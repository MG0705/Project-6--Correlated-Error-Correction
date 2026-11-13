#This python file contains the unit tests for the differentiable mesh, error models, kriging and correction in mesh_error_correction.py
import numpy as np
import pytest
from src.mesh_error_correction import (random_unitary, decompose, rebuild, fidelity, angle_to_power, MeshModel, pack_phases, correlation, field_factor, sample_errors, choose_monitors, krige, run_chip, sigma_sweep, corr_length_sweep, monitor_sweep)


# -----------------------------
# Fixtures
# -----------------------------

#fixture for a random number generator with a fixed seed, so that the tests are reproducible
@pytest.fixture
def rng():
    return np.random.default_rng(42)


#fixture for a compiled 5-mode mesh model
@pytest.fixture(scope="module")
def model():
    return MeshModel(5)

#fixture from a random target unitary, its ideal phases and one fabricated set of coupler splits
@pytest.fixture
def chip(model, rng):
    U = random_unitary(5, rng)
    mzis, d = decompose(U)
    a = sample_errors("iid", 0.05, 2 * model.n, rng)
    return U, mzis, d, pack_phases(mzis, np.angle(d)), angle_to_power(a)


# -----------------------------
# Unit Tests: differentiable mesh model
# -----------------------------

#the JAX forward model should give exactly the matrix that the numpy rebuild() gives, with coupler errors
def test_jax_model_matches_rebuild(model, chip):
    _, mzis, d, p0, splits = chip
    couplers = [(splits[2 * k], splits[2 * k + 1]) for k in range(len(mzis))]
    assert np.allclose(np.asarray(model.forward(p0, splits)), rebuild(mzis, np.angle(d), couplers), atol=1e-12)

#every unitary of a given size compiles to the same MZI layout, so one compiled model serves all targets
def test_layout_is_fixed_for_a_given_size(model, rng):
    mzis, _ = decompose(random_unitary(5, rng))
    assert tuple(z.m for z in mzis) == model.ms
    assert model.positions().shape == (2 * model.n, 2)


#the model's 1-F should equal the fidelity() from the shared mesh code, for a random target and one fabricated chip
def test_infidelity_matches_fidelity(model, chip):
    U, mzis, d, p0, splits = chip
    couplers = [(splits[2 * k], splits[2 * k + 1]) for k in range(len(mzis))]
    assert model.infidelity(p0, splits, U) == pytest.approx(1 - fidelity(U, rebuild(mzis, np.angle(d), couplers)), abs=1e-12)


#with perfect couplers the compiled phases already give zero infidelity, and correction keeps it there
def test_ideal_chip_needs_no_correction(model, chip):
    U, _, _, p0, _ = chip
    ideal = np.full(2 * model.n, 0.5)
    assert model.infidelity(p0, ideal, U) == pytest.approx(0.0, abs=1e-12)
    assert model.infidelity(model.correct(p0, ideal, U), ideal, U) == pytest.approx(0.0, abs=1e-12)

#reoptimising the phases for the true couplers should cut the infidelity by a large factor
def test_exact_correction_reduces_infidelity(model, chip):
    U, _, _, p0, splits = chip
    before = model.infidelity(p0, splits, U)
    after = model.infidelity(model.correct(p0, splits, U), splits, U)
    assert after < before / 100


# -----------------------------
# Unit Tests: fabrication error and monitors
# -----------------------------

#the correlated field should have per-coupler RMS equal to sigma (within 10%)
def test_field_has_requested_rms(model, rng):
    L = field_factor(model.positions(), 2.0)
    draws = np.array([sample_errors("field", 0.03, 2 * model.n, rng, L) for _ in range(2000)])
    assert np.sqrt((draws ** 2).mean()) == pytest.approx(0.03, rel=0.1)


def test_error_model_errors(rng):
    with pytest.raises(ValueError):
        sample_errors("field", 0.05, 10, rng)
    with pytest.raises(ValueError):
        sample_errors("not-a-model", 0.05, 10, rng)
    with pytest.raises(ValueError):
        choose_monitors(10, 0.0, rng)


def test_choose_monitors(rng):
    mon = choose_monitors(40, 0.25, rng)
    assert len(mon) == 10 and len(np.unique(mon)) == 10


# -----------------------------
# Unit Tests: kriging
# -----------------------------

#measuring every coupler with tiny noise should return the true errors
def test_kriging_with_every_coupler_monitored_recovers_errors(model, rng):
    pos = model.positions()
    C = correlation(pos, 2.0)
    a = sample_errors("field", 0.05, len(pos), rng, field_factor(pos, 2.0))
    mon = np.arange(len(pos))
    assert np.allclose(krige(a, mon, C, 0.05, 1e-6), a, atol=1e-5)

#with independent errors, measuring some couplers says nothing about the others, so their estimate is zero
def test_kriging_cannot_predict_independent_errors(model, rng):
    n = 2 * model.n
    a = sample_errors("iid", 0.05, n, rng)
    mon = choose_monitors(n, 0.5, rng)
    est = krige(a[mon], mon, np.eye(n), 0.05, 0.002)
    assert np.allclose(np.delete(est, mon), 0.0)


def test_kriging_helps_for_a_smooth_field(model, rng):
    pos = model.positions()
    L = field_factor(pos, 8.0)
    mon = choose_monitors(len(pos), 0.25, rng)
    err, prior = [], []
    for i in range(50):
        a = sample_errors("field", 0.05, len(pos), rng, L)
        err.append(np.mean((krige(a[mon], mon, correlation(pos, 8.0), 0.05, 0.002) - a) ** 2))
        prior.append(np.mean(a ** 2))
    assert np.mean(err) < 0.1 * np.mean(prior)


# -----------------------------
# Unit Tests: experiments
# -----------------------------

#one chip returns a 1-F for every method plus the estimate error, and exact correction is the best
def test_run_chip_returns_every_method(model, rng):
    r = run_chip(model, "iid", 0.05, None, 0.25, rng)
    assert set(r) == {"uncorrected", "monitored", "exact", "estimate rms"}
    assert r["exact"] <= r["monitored"] + 1e-12 and r["exact"] < r["uncorrected"]

#small sweeps return one result per setting, each with a mean and a standard error
def test_experiment_shapes():
    res = sigma_sweep(N=3, sigmas=(0.02, 0.05), n_chips=2, verbose=False)
    assert len(res) == 2 and all(len(points) == 2 for points in res.values())
    res = corr_length_sweep(N=3, corr_lengths=(1, 4), n_chips=2, verbose=False)
    assert len(res["field"]) == 2 and len(res["independent"]["exact"]) == 2
    res = monitor_sweep(N=3, fractions=(0.5, 1.0), n_chips=2, verbose=False)
    assert len(res["monitored"]) == 2 and set(res["reference"]) == {"uncorrected", "exact"}
