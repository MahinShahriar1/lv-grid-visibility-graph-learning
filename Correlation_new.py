import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ======================
# CONFIG
# ======================
RANDOM_STATE = 0
TRAIN_FRACTION = 0.7
DROP_TIME_COL_IF_PRESENT = True

DOWNSAMPLE_FACTOR = 1      # set >1 if you want faster (e.g., 5)
MAX_TIME_SAMPLES = None    # set int if you want cap (e.g., 50000)
TIME_SAMPLING = "linspace" # "linspace" or "random"

EPS = 1e-12

# ======================
# Gaussian-copula MI helpers
# ======================
def _rank_to_uniform(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)

    xs = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        r = 0.5 * (i + j) + 1.0
        ranks[order[i:j+1]] = r
        i = j + 1

    u = (ranks - 0.5) / n
    return np.clip(u, EPS, 1.0 - EPS)

def _norm_ppf_approx(u: np.ndarray) -> np.ndarray:
    a = [-3.969683028665376e+01,  2.209460984245205e+02, -2.759285104469687e+02,
          1.383577518672690e+02, -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02, -1.556989798598866e+02,
          6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00,  4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,  2.445134137142996e+00,
          3.754408661907416e+00]

    u = np.asarray(u, dtype=np.float64)
    plow = 0.02425
    phigh = 1 - plow
    x = np.empty_like(u)

    m = u < plow
    if np.any(m):
        q = np.sqrt(-2 * np.log(u[m]))
        x[m] = (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    m = (u >= plow) & (u <= phigh)
    if np.any(m):
        q = u[m] - 0.5
        r = q*q
        x[m] = (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
               (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)

    m = u > phigh
    if np.any(m):
        q = np.sqrt(-2 * np.log(1 - u[m]))
        x[m] = -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)

    return x

def gaussian_copula_mi_weights(X: np.ndarray) -> np.ndarray:
    """
    Returns a single matrix W in [0,1] to use as MI-weight adjacency.
    """
    T, N = X.shape

    # rank->uniform->gaussianize per node
    Z = np.empty((T, N), dtype=np.float64)
    for j in range(N):
        u = _rank_to_uniform(X[:, j])
        Z[:, j] = _norm_ppf_approx(u)

    # correlation on Z
    Z -= Z.mean(axis=0, keepdims=True)
    std = Z.std(axis=0, ddof=1)
    std[std < EPS] = 1.0
    Z /= std

    R = (Z.T @ Z) / (T - 1)
    R = np.clip(R, -0.999999, 0.999999)

    # Gaussian MI on copula-Gaussianized data
    MI = -0.5 * np.log(1.0 - R**2 + EPS)
    np.fill_diagonal(MI, 0.0)

    # Robust scale to [0,1] using 99th percentile (not true NMI, just weights)
    iu = np.triu_indices(N, 1)
    s = float(np.quantile(MI[iu], 0.99))
    s = max(s, EPS)

    W = np.clip(MI / s, 0.0, 1.0)
    np.fill_diagonal(W, 0.0)
    return W

# ======================
# MAIN (per phase)
# ======================
def compute_phase(phase: str, data_dir="."):
    nodes_path = os.path.join(data_dir, f"nodesMVLV_phase{phase}.csv")
    vmag_path  = os.path.join(data_dir, f"MVLV_VmagTure_phase{phase}.csv")

    out_w = os.path.join(data_dir, f"MI_weight_phase{phase}.csv")

    nodes = pd.read_csv(nodes_path, header=None).iloc[0].to_list()
    N = len(nodes)

    df = pd.read_csv(vmag_path, header=None)
    if DROP_TIME_COL_IF_PRESENT and df.shape[1] == N + 1:
        df = df.iloc[:, 1:]  # drop time col if present
    df.columns = [str(x) for x in nodes]

    X_full = df.values
    X_train, _ = train_test_split(X_full, test_size=(1 - TRAIN_FRACTION), shuffle=False)

    if DOWNSAMPLE_FACTOR and DOWNSAMPLE_FACTOR > 1:
        X_train = X_train[::DOWNSAMPLE_FACTOR, :]

    if MAX_TIME_SAMPLES is not None and X_train.shape[0] > MAX_TIME_SAMPLES:
        T = X_train.shape[0]
        if TIME_SAMPLING == "random":
            rng = np.random.default_rng(RANDOM_STATE)
            idx = rng.choice(T, size=MAX_TIME_SAMPLES, replace=False)
            idx.sort()
        else:
            idx = np.linspace(0, T - 1, MAX_TIME_SAMPLES, dtype=int)
        X_train = X_train[idx, :]

    print(f"[Phase {phase}] Train samples x nodes = {X_train.shape}")

    W = gaussian_copula_mi_weights(X_train)

    pd.DataFrame(W).to_csv(out_w, index=False, header=False)
    print(f"[Phase {phase}] Saved MI-weight adjacency (0..1): {out_w}")

if __name__ == "__main__":
    DATA_DIR = "."
    for ph in ["A", "B", "C"]:
        compute_phase(ph, DATA_DIR)
