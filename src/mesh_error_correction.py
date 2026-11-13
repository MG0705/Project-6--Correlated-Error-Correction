### Correcting spatially correlated fabrication error in programmable photonic meshes from sparse monitors ###
# A target unitary is compiled onto a Clements mesh whose couplers carry fabrication error (independent, or a
# spatially correlated field). Three ways of setting the phases are compared:
#   uncorrected - the phases compiled for an ideal chip
#   exact       - phases re-optimised knowing every coupler's true error (the best correction can do)
#   monitored   - only a fraction of couplers are measured (with noise); the rest are estimated by Gaussian-process
#                 regression (kriging) using the spatial correlation, and the phases are re-optimised for the estimate
# The re-optimisation uses a differentiable JAX model of the mesh and L-BFGS.

import argparse
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from dataclasses import dataclass
from scipy.optimize import minimize
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp


###[1] Core unitary matrix decomposition, rebuild and fidelity in Clements' rectangular mesh ###
@dataclass
class MZI:
    layer: int
    m: int
    theta: float
    phi: float

#The 50:50 beam splitter matrix, generalised to an arbitrary power splitting ratio
#0.5 is ideal and the minus sign in the lower right is what makes the columns orthogonal - without it the device would not be
#unitary and would not conserve energy
def beamsplitter(power_split=0.5):
    t, k = np.sqrt(1.0-power_split), np.sqrt(power_split)
    return np.array([[t, k], [k, -t]], dtype=complex)


#Matrix representation of a single MZI with phase shifters theta and phi
#Alpha and beta are the power splitting ratios of the two beam splitters
#They default to ideal, so mzi_matrix(theta, phi) is the same as before
def mzi_matrix(theta, phi, alpha=0.5, beta=0.5):
    phase_ext = np.array([[np.exp(1j*theta), 0], [0, 1]], dtype=complex) #the phase shifter matrix (by theta)
    phase_int = np.array([[np.exp(1j*phi), 0], [0, 1]], dtype=complex) #the internal phase shifter matrix (by phi)
    return beamsplitter(beta) @ phase_int @ beamsplitter(alpha) @ phase_ext


#Embedding M into a NxN with 1s on the diagonal and 0s everywhere else to give matrix T
def embed(M,m,N): #M is 2x2 MZI matrix, m is column and row index, N is new dimension
    T=np.eye(N, dtype=complex) #identity matrix of size N
    T[m:m+2, m:m+2] = M
    return T

#Haar-random unitary and the normalisation of the R diagonal is required - without it the result is biased and not actually Haar distributed
def random_unitary(N, rng):
    z = rng.normal(size=(N,N)) + 1j*rng.normal(size=(N,N))
    q, r = np.linalg.qr(z)
    return q @ np.diag(np.diag(r)/np.abs(np.diag(r)))


#Assigns the correct layer to each MZI based on the order of operations and the number of MZIs in each layer
def assign_layers(ops, N):
    free = [0]*N
    mzis = []
    for (m, theta, phi) in reversed(ops):
        layer = max(free[m], free[m+1])
        while layer % 2 != m % 2:
            layer += 1
        free[m] = free[m+1] = layer + 1
        mzis.append(MZI(layer=layer, m=m,
                        theta=float(theta % (2*np.pi)),
                        phi=float(phi % (2*np.pi))))
    return mzis


#decomposing U into a product of MZIs and phase shifters by zeroing as shown below
#following U = [* * * *]
#              [6 * * *]
#              [2 5 * *]
#              [1 3 4 *]

def decompose(U, verbose=False):
    N=U.shape[0]
    U = U.astype(complex).copy()
    left_ops,right_ops = [],[]

    def null_by_col(U,row,m) -> tuple[float, float]:
        if U[row,m+1] == 0:
            return float(np.angle(U[row,m]) - np.pi/2), float(np.pi)

        ratio = U[row,m]/U[row,m+1]
        theta = float(np.angle(ratio) - np.pi/2)
        phi = float(2*np.arctan(np.abs(ratio)))
        return theta,phi

    def null_by_row(U,col,m) -> tuple[float, float]:
        if U[m,col] == 0:
            return float(np.angle(U[m+1,col]) + np.pi/2), float(np.pi)

        ratio = U[m+1,col]/U[m,col]
        theta = float(np.angle(ratio) + np.pi/2)
        phi = float(2*np.arctan(np.abs(ratio)))
        return theta,phi

    def diagonal_D(m,theta,phi,d) -> np.ndarray:#takes the final D obtained from U and turns it into D_new using a lemma
        d1, d2 = d[m], d[m+1]
        theta_new = float(np.angle(-d1 / d2))
        common = np.exp(-1j*phi)
        d[m] = -common * np.exp(-1j*theta) * d2
        d[m+1] = common * d2
        return theta_new,d

    for i in range(1,N): #indexing used here is from 1 to more easily follow matrix subscripts
        if i%2 != 0:
            for j in range(0,i):
                m=i-j-1
                theta, phi = null_by_col(U,N-j-1,m)
                M = mzi_matrix(theta,phi)
                mat_T = embed(M,m,N)
                U =  U @ (mat_T.conj().T)
                right_ops.append((m, theta, phi))
                if verbose:
                    print(f"stripe {i}: zeroed U[{N-j-1},{m}]"); print(np.round(np.abs(U),3), "\n")
        else:
            for j in range(1,i+1):
                m= N+j-i-2
                theta, phi = null_by_row(U,j-1,m)
                M = mzi_matrix(theta,phi)
                mat_T = embed(M,m,N)
                U = mat_T @ U
                left_ops.append((m, theta, phi))
                if verbose:
                    print(f"stripe {i}: zeroed U[{m+1},{j-1}]"); print(np.round(np.abs(U),3), "\n")

    d = np.diag(U).copy()
    commuted = []
    for (m, theta, phi) in reversed(left_ops):
        theta_new, d = diagonal_D(m, theta, phi, d)
        commuted.append((m, theta_new, phi))

    operations = list(reversed(commuted)) + list(reversed(right_ops))
    return assign_layers(operations, N), d

#couplers is an optional list of (alpha, beta) pairs, one per MZI in the same order as mzis
# Coupler error is a property of the fabricated device rather than a knob setting, which is why it lives here and not in the MZI dataclass
def rebuild(mzis, output_phases, couplers=None):
    N = len(output_phases)
    X = np.diag(np.exp(1j*np.asarray(output_phases)))
    for k in range(len(mzis)-1, -1, -1):
        item = mzis[k]
        if couplers is None:
            M = mzi_matrix(item.theta, item.phi)
        else:
            alpha, beta = couplers[k]
            M = mzi_matrix(item.theta, item.phi, alpha, beta)
        X = X @ embed(M, item.m, N)
    return X

#normalised overlap between two unitary matrices, which is a measure of how close they are to each other
def fidelity(U,V):
    return float(abs(np.trace(U.conj().T @ V))/U.shape[0])


###[2] Error conversions###
#Power splitting ratio as per Bandyopadhyay's angular deviation model
def power_to_angle(p):
    return np.arcsin(np.sqrt(np.clip(p, 0.0, 1.0))) -np.pi/4

#Angular deviation converted to power splitting ratio as per Bandyopadhyay's angular deviation model
def angle_to_power(a):
    return np.sin(np.asarray(a)+ np.pi/4)**2



###[3] Differentiable mesh model (JAX)###
#Same conventions as mzi_matrix() and rebuild() above with X = D T_{n-1} ... T_0, where each T_k acts on modes (m_k, m_k+1).
#Phases are packed as an array of [theta_0..theta_{n-1}, phi_0..phi_{n-1}, output_0..output_{N-1}].
def _bs(p):
    t, k = jnp.sqrt(1.0 - p), jnp.sqrt(p)
    return jnp.array([[t, k], [k, -t]], dtype=jnp.complex128)

def _mzi(theta, phi, alpha, beta):
    ext = jnp.array([[jnp.exp(1j * theta), 0], [0, 1]], dtype=jnp.complex128)
    inn = jnp.array([[jnp.exp(1j * phi), 0], [0, 1]], dtype=jnp.complex128)
    return _bs(beta) @ inn @ _bs(alpha) @ ext

def pack_phases(mzis, output_phases):
    return np.concatenate([[z.theta for z in mzis], [z.phi for z in mzis], np.asarray(output_phases, dtype=float)])

#The mesh layout (which modes each MZI acts on) is fixed for a given N, so the model is built and compiled once per N
class MeshModel:
    def __init__(self, N, rng=None):
        rng = np.random.default_rng(0) if rng is None else rng
        mzis, _ = decompose(random_unitary(N, rng)) #any unitary gives the same layout
        self.N, self.ms, self.n = N, tuple(z.m for z in mzis), len(mzis)
        self.layers = tuple(z.layer for z in mzis)

        def forward(params, splits):
            n = self.n
            X = jnp.diag(jnp.exp(1j * params[2 * n:]))
            for k in range(n - 1, -1, -1):
                m = self.ms[k]
                X = X.at[:, m:m + 2].set(X[:, m:m + 2] @ _mzi(params[k], params[n + k], splits[2 * k], splits[2 * k + 1]))
            return X

        def loss(params, splits, U):
            return 1.0 - jnp.abs(jnp.trace(U.conj().T @ forward(params, splits))) / N

        self.forward = jax.jit(forward)
        self._loss_grad = jax.jit(jax.value_and_grad(loss))

    #positions (x, y) of every coupler in coupler pitches, two per MZI, in the same order as the splits vector
    def positions(self):
        return np.array([(2 * l + j, m + 0.5) for l, m in zip(self.layers, self.ms) for j in (0, 1)], dtype=float)

    def infidelity(self, params, splits, U):
        return float(self._loss_grad(jnp.asarray(params), jnp.asarray(splits), jnp.asarray(U))[0])

    #re-optimise every phase so that the mesh, with the given coupler splits, best matches U
    def correct(self, params0, splits, U, max_iter=500):
        splits, U = jnp.asarray(splits), jnp.asarray(U)
        def f(p):
            v, g = self._loss_grad(jnp.asarray(p), splits, U)
            return float(v), np.asarray(g, dtype=float)
        res = minimize(f, np.asarray(params0, dtype=float), jac=True, method="L-BFGS-B", options={"maxiter": max_iter, "ftol": 1e-15, "gtol": 1e-12})
        return res.x


###[4] Fabrication error and sparse monitors###
#Cholesky factor of the unit-variance squared-exponential correlation exp(-d^2 / 2 corr_length^2)
def correlation(pos, corr_length):
    d2 = ((pos[:, None, :] - pos[None, :, :]) ** 2).sum(-1)
    return np.exp(-d2 / (2 * corr_length ** 2))

def field_factor(pos, corr_length):
    return np.linalg.cholesky(correlation(pos, corr_length) + 1e-10 * np.eye(len(pos)))

#angular deviation (rad) of every coupler: "iid", or "field" (pass the Cholesky factor L)
def sample_errors(kind, sigma, n, rng, L=None):
    if kind == "iid":
        return rng.normal(0, sigma, n)
    if kind == "field":
        if L is None:
            raise ValueError("the field model needs a Cholesky factor from field_factor()")
        return sigma * (L @ rng.normal(size=n))
    raise ValueError(f"unknown error model: {kind}")

#indices of the monitored couplers are a fixed, spatially random fraction of the coupler sites
def choose_monitors(n, fraction, rng):
    k = int(round(fraction * n))
    if not 1 <= k <= n:
        raise ValueError("the monitor fraction must select between 1 and all of the couplers")
    return np.sort(rng.choice(n, size=k, replace=False))

#Gaussian-process (kriging) estimate of every coupler error from noisy measurements y of the monitored ones.
#prior covariance sigma^2 C, where C is the identity for independent errors or the field correlation otherwise.
def krige(y, monitors, C, sigma, sigma_meas):
    K_ss = sigma ** 2 * C[np.ix_(monitors, monitors)] + sigma_meas ** 2 * np.eye(len(monitors))
    K_as = sigma ** 2 * C[:, monitors]
    return K_as @ np.linalg.solve(K_ss, y)


###[5] Experiments###
SIGMA_MEAS = 0.002 #RMS error of a single coupler measurement

#one fabricated chip and one random target with the infidelity for every method, plus the RMS error of the estimate
def run_chip(model, kind, sigma, corr_length, fraction, rng, C=None, L=None, sigma_meas=SIGMA_MEAS, methods=None):
    methods = ("uncorrected", "monitored", "exact") if methods is None else methods
    N = model.N
    U = random_unitary(N, rng)
    mzis, d = decompose(U)
    p0 = pack_phases(mzis, np.angle(d))
    pos = model.positions()
    a = sample_errors(kind, sigma, len(pos), rng, L)
    true = angle_to_power(a)
    out = {}
    if "uncorrected" in methods:
        out["uncorrected"] = model.infidelity(p0, true, U)
    if "monitored" in methods:
        mon = choose_monitors(len(pos), fraction, rng)
        Cm = np.eye(len(pos)) if kind == "iid" else C
        a_hat = krige(a[mon] + rng.normal(0, sigma_meas, len(mon)), mon, Cm, sigma, sigma_meas)
        out["monitored"] = model.infidelity(model.correct(p0, angle_to_power(a_hat), U), true, U)
        out["estimate rms"] = float(np.sqrt(np.mean((a_hat - a) ** 2)))
    if "exact" in methods:
        out["exact"] = model.infidelity(model.correct(p0, true, U), true, U)
    return out

#mean and standard error of each method's infidelity over the num of chips
def mean_over_chips(model, kind, sigma, corr_length, fraction, rng, n_chips, methods=None):
    pos = model.positions()
    C = correlation(pos, corr_length) if kind == "field" else None
    L = field_factor(pos, corr_length) if kind == "field" else None
    runs = [run_chip(model, kind, sigma, corr_length, fraction, rng, C, L, methods=methods) for _ in range(n_chips)]
    return {key: (float(np.mean([r[key] for r in runs])), float(np.std([r[key] for r in runs], ddof=1) / np.sqrt(n_chips)))
            for key in runs[0]}

ERROR_MODELS = {"independent": ("iid", None), "correlated (corr_length=2)": ("field", 2.0)}

#Experiment 1: infidelity against coupler error, for each error model and correction method (with 25% of couplers monitored)
def sigma_sweep(N=8, sigmas=(0.01, 0.02, 0.05, 0.1), fraction=0.25, n_chips=40, seed=1, verbose=True):
    rng = np.random.default_rng(seed)
    model = MeshModel(N)
    out = {}
    for name, (kind, ell) in ERROR_MODELS.items():
        out[name] = [mean_over_chips(model, kind, s, ell, fraction, rng, n_chips) for s in sigmas]
        if verbose:
            print(f"  {name} done")
    return out

#Experiment 2: infidelity against correlation length at fixed coupler error, with independent errors as reference
def corr_length_sweep(N=8, sigma=0.05, corr_lengths=(0.5, 1, 2, 4, 8, 16), fraction=0.25, n_chips=40, seed=2, verbose=True):
    rng = np.random.default_rng(seed)
    model = MeshModel(N)
    out = {"corr_lengths": np.asarray(corr_lengths, dtype=float),
           "independent": mean_over_chips(model, "iid", sigma, None, fraction, rng, n_chips)}
    out["field"] = []
    for ell in corr_lengths:
        out["field"].append(mean_over_chips(model, "field", sigma, ell, fraction, rng, n_chips))
        if verbose:
            print(f"  corr_length = {ell:g} done")
    return out

#Experiment 3: how many monitors are needed with infidelity against the monitored fraction for a correlated chip
def monitor_sweep(N=8, sigma=0.05, corr_length=2.0, fractions=(0.05, 0.1, 0.25, 0.5, 1.0), n_chips=40, seed=3, verbose=True):
    rng = np.random.default_rng(seed)
    model = MeshModel(N)
    out = {"fractions": np.asarray(fractions, dtype=float), "reference": mean_over_chips(model, "field", sigma, corr_length, 1.0, rng, n_chips, ("uncorrected", "exact"))}
    out["monitored"] = []
    for f in fractions:
        out["monitored"].append(mean_over_chips(model, "field", sigma, corr_length, f, rng, n_chips, ("monitored",)))
        if verbose:
            print(f"  fraction = {f:g} done")
    return out

def series(points, key):
    return np.array([p[key][0] for p in points]), np.array([p[key][1] for p in points])


###[6] Plotting, saving and command-line interface###
METHOD_STYLE = {"uncorrected": ("C3", "uncorrected (ideal-chip phases)"), "monitored": ("C0", "corrected from 25% monitors + kriging"), "exact": ("C2", "corrected with every coupler known")}

def _band(x, mean, se, **kw):
    line, = plt.plot(x, mean, marker="o", **kw)
    plt.fill_between(x, np.maximum(mean - se, mean * 0.1), mean + se, color=line.get_color(), alpha=0.15)

def plot_sigma_sweep(res, sigmas, N, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for ax, (name, points) in zip(axes, res.items()):
        plt.sca(ax)
        for method, (color, label) in METHOD_STYLE.items():
            _band(sigmas, *series(points, method), color=color, label=label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks(sigmas, [f"{s:g}" for s in sigmas])
        ax.minorticks_off()
        ax.set_xlabel("RMS coupler angular deviation (rad, log scale)")
        ax.set_title(f"{name} coupler error")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("infidelity 1 - |Tr(U^dagger V)| / N")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{N}x{N} Clements mesh: infidelity before and after correction")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def plot_corr_length_sweep(res, sigma, N, path):
    plt.figure(figsize=(7.5, 5))
    ells = res["corr_lengths"]
    for method, (color, label) in METHOD_STYLE.items():
        _band(ells, *series(res["field"], method), color=color, label=label + ", correlated")
        plt.axhline(res["independent"][method][0], color=color, ls="--", lw=1)
    plt.plot([], [], color="0.4", ls="--", lw=1, label="same method, independent errors")
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(ells, [f"{e:g}" for e in ells])
    plt.minorticks_off()
    plt.xlabel("correlation length (coupler pitches, log scale)")
    plt.ylabel("infidelity 1 - |Tr(U^dagger V)| / N")
    plt.title(f"{N}x{N} mesh, sigma = {sigma} rad: correlation makes sparse monitoring work")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

def plot_monitor_sweep(res, sigma, N, path):
    plt.figure(figsize=(7.5, 5))
    fr = res["fractions"] * 100
    _band(fr, *series(res["monitored"], "monitored"), color="C0", label="corrected from monitors + kriging")
    for method in ("uncorrected", "exact"):
        color, label = METHOD_STYLE[method]
        plt.axhline(res["reference"][method][0], color=color, ls="--", label=label)
    plt.xscale("log")
    plt.yscale("log")
    plt.xticks(fr, [f"{f:g}" for f in fr])
    plt.minorticks_off()
    plt.xlabel("couplers monitored (%)")
    plt.ylabel("infidelity 1 - |Tr(U^dagger V)| / N")
    plt.title(f"{N}x{N} mesh, sigma = {sigma} rad, corr_length = 2: how many monitors are needed")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()

KEYS = ("uncorrected", "monitored", "exact", "estimate rms")

def save_sigma_csv(res, sigmas, path):
    with open(path, "w") as f:
        f.write("error_model,sigma_rad," + ",".join(f"{k.replace(' ', '_')},{k.replace(' ', '_')}_se" for k in KEYS) + "\n")
        for name, points in res.items():
            for s, p in zip(sigmas, points):
                f.write(f"{name},{s}," + ",".join(f"{p[k][0]:.4e},{p[k][1]:.1e}" for k in KEYS) + "\n")

def save_corr_csv(res, path):
    with open(path, "w") as f:
        f.write("corr_length," + ",".join(f"{k.replace(' ', '_')},{k.replace(' ', '_')}_se" for k in KEYS) + "\n")
        for ell, p in list(zip(res["corr_lengths"], res["field"])) + [("independent", res["independent"])]:
            f.write(f"{ell}," + ",".join(f"{p[k][0]:.4e},{p[k][1]:.1e}" for k in KEYS) + "\n")

def save_monitor_csv(res, path):
    with open(path, "w") as f:
        f.write("monitored_fraction,monitored,monitored_se,estimate_rms,estimate_rms_se\n")
        for fr, p in zip(res["fractions"], res["monitored"]):
            f.write(f"{fr:g},{p['monitored'][0]:.4e},{p['monitored'][1]:.1e},{p['estimate rms'][0]:.4e},{p['estimate rms'][1]:.1e}\n")
        for method in ("uncorrected", "exact"):
            f.write(f"reference_{method},{res['reference'][method][0]:.4e},{res['reference'][method][1]:.1e},,\n")

def main():
    parser = argparse.ArgumentParser(description="Correcting correlated fabrication error in photonic meshes from sparse monitors")
    parser.add_argument("--experiment", choices=["sigma", "corr_length", "monitors", "all"], default="all")
    parser.add_argument("--modes", type=int, default=8, help="mesh size N")
    parser.add_argument("--chips", type=int, default=40, help="fabricated chips (each with a new random target) per point")
    parser.add_argument("--sigma", type=float, default=0.05, help="coupler error for Experiments 2 and 3 (rad)")
    parser.add_argument("--fraction", type=float, default=0.25, help="fraction of couplers monitored in Experiments 1 and 2")
    parser.add_argument("--outdir", default="figures")
    parser.add_argument("--resultsdir", default="results")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(args.resultsdir, exist_ok=True)
    N = args.modes
    METHOD_STYLE["monitored"] = ("C0", f"corrected from {args.fraction:.0%} monitors + kriging")

    print("=== Correcting Correlated Fabrication Error from Sparse Monitors ===")
    print(f"{N}x{N} Clements mesh | {args.chips} chips per point | monitor noise {SIGMA_MEAS} rad\n")

    if args.experiment in ("sigma", "all"):
        sigmas = (0.01, 0.02, 0.05, 0.1)
        print(f"Experiment 1: infidelity vs coupler error ({args.fraction:.0%} of couplers monitored)")
        res = sigma_sweep(N, sigmas, args.fraction, args.chips)
        for name, points in res.items():
            print(f"\n{name}\n{'sigma':>7s} {'uncorrected':>12s} {'monitored':>12s} {'exact':>12s} {'estimate rms / sigma':>21s}")
            for s, p in zip(sigmas, points):
                print(f"{s:7g} {p['uncorrected'][0]:12.2e} {p['monitored'][0]:12.2e} {p['exact'][0]:12.2e} {p['estimate rms'][0] / s:21.2f}")
        plot_sigma_sweep(res, sigmas, N, os.path.join(args.outdir, "infidelity_vs_sigma.png"))
        save_sigma_csv(res, sigmas, os.path.join(args.resultsdir, "sigma_sweep.csv"))
        print(f"saved {args.outdir}/infidelity_vs_sigma.png and {args.resultsdir}/sigma_sweep.csv\n")

    if args.experiment in ("corr_length", "all"):
        print(f"Experiment 2: infidelity vs correlation length (sigma = {args.sigma} rad, {args.fraction:.0%} monitored)")
        res = corr_length_sweep(N, args.sigma, fraction=args.fraction, n_chips=args.chips)
        print(f"\n{'corr_length':>12s} {'uncorrected':>12s} {'monitored':>12s} {'exact':>12s} {'estimate rms / sigma':>21s}")
        for ell, p in list(zip(res["corr_lengths"], res["field"])) + [("independent", res["independent"])]:
            label = ell if isinstance(ell, str) else f"{ell:g}"
            print(f"{label:>12s} {p['uncorrected'][0]:12.2e} {p['monitored'][0]:12.2e} {p['exact'][0]:12.2e} {p['estimate rms'][0] / args.sigma:21.2f}")
        plot_corr_length_sweep(res, args.sigma, N, os.path.join(args.outdir, "infidelity_vs_corr_length.png"))
        save_corr_csv(res, os.path.join(args.resultsdir, "corr_length_sweep.csv"))
        print(f"saved {args.outdir}/infidelity_vs_corr_length.png and {args.resultsdir}/corr_length_sweep.csv\n")

    if args.experiment in ("monitors", "all"):
        print(f"Experiment 3: infidelity vs fraction of couplers monitored (sigma = {args.sigma} rad, corr_length = 2)")
        res = monitor_sweep(N, args.sigma, n_chips=args.chips)
        print(f"\n{'monitored':>10s} {'infidelity':>12s} {'estimate rms / sigma':>21s}")
        for fr, p in zip(res["fractions"], res["monitored"]):
            print(f"{fr:10.0%} {p['monitored'][0]:12.2e} {p['estimate rms'][0] / args.sigma:21.2f}")
        print(f"{'none':>10s} {res['reference']['uncorrected'][0]:12.2e}   (uncorrected)")
        print(f"{'exact':>10s} {res['reference']['exact'][0]:12.2e}   (every coupler known exactly)")
        plot_monitor_sweep(res, args.sigma, N, os.path.join(args.outdir, "infidelity_vs_monitors.png"))
        save_monitor_csv(res, os.path.join(args.resultsdir, "monitor_sweep.csv"))
        print(f"saved {args.outdir}/infidelity_vs_monitors.png and {args.resultsdir}/monitor_sweep.csv")


if __name__ == "__main__":
    main()
