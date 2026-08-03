import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import random
from torch_geometric.nn import GCNConv
from sklearn.metrics import mean_absolute_error, mean_squared_error
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from pathlib import Path


# ============================================================
# OUTPUT FOLDER (MATCH Evaluation_random)
# ============================================================
SAVED_TIMESERIES_DIR = Path(r"D:\Python Programming\Advanced Project of Smart Meter\saved_timeseries")
SAVED_TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Fonts (optional)
# ------------------------------------------------------------
try:
    font_path = r"C:\Users\Mahin Shahriar\AppData\Local\Microsoft\Windows\Fonts\LinLibertine_Rah.ttf"
    fm.fontManager.addfont(font_path)
    plt.rcParams.update({"font.family": ["Linux Libertine"]})
except Exception:
    pass

# --- NEW IMPORT FOR GBR SPLIT (kept as in your file) ---
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Warning: sklearn.model_selection not found. Calibrator Early Stopping will be disabled.")
    train_test_split = None


# ------------------------------------------------------------
# Loader
# ------------------------------------------------------------
try:
    from Load_GCN import StaticVoltageLoader  # type: ignore
except ImportError:
    print("=" * 50)
    print("❌ ERROR: Could not find 'Loader_File2.py'")
    print("Please ensure StaticVoltageLoader is available.")
    print("=" * 50)
    raise

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"--- Using device: {device} for Evaluation ---")


# ============================================================
# Metrics helpers
# ============================================================
def max_ae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.max(np.abs(y_true - y_pred)))


def nrmse(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse_val = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = float(y_true.max() - y_true.min())
    if denom < eps:
        return np.nan
    return float(rmse_val / denom)


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num = np.abs(y_true - y_pred)
    denom = np.clip((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps, None)
    return float(np.mean(num / denom) * 100.0)


def per_node_metrics(y_true_2d, y_pred_2d):
    """Column-wise metrics for 2D arrays [T, N]. Returns vectors length N."""
    err = y_true_2d - y_pred_2d
    abs_err = np.abs(err)

    mae = abs_err.mean(axis=0)
    rmse_vec = np.sqrt((err ** 2).mean(axis=0))
    maxae = abs_err.max(axis=0)

    y_range = (y_true_2d.max(axis=0) - y_true_2d.min(axis=0))
    nrmse_vec = np.where(y_range > 1e-8, rmse_vec / y_range, np.nan)

    denom = np.clip(np.abs(y_true_2d), 1e-8, None)
    mape_vec = (np.abs(err) / denom).mean(axis=0) * 100.0

    denom2 = np.clip((np.abs(y_true_2d) + np.abs(y_pred_2d)) / 2.0, 1e-8, None)
    smape_vec = (abs_err / denom2).mean(axis=0) * 100.0

    return mae, rmse_vec, maxae, nrmse_vec, mape_vec, smape_vec


# ============================================================
# FULL TEST SCALAR METRICS (MATCH Evaluation_random)
# ============================================================
def fullset_scalar_metrics(y_true_2d, y_pred_2d):
    """
    returns dict with MAE, RMSE, Max_TW_AE, Max_NW_AE (same as Evaluation_random)
    """
    yt = np.asarray(y_true_2d)
    yp = np.asarray(y_pred_2d)

    # tolerate [T,N,1]
    if yt.ndim == 3 and yt.shape[-1] == 1:
        yt = yt[:, :, 0]
    if yp.ndim == 3 and yp.shape[-1] == 1:
        yp = yp[:, :, 0]

    if yt.shape != yp.shape:
        raise ValueError(f"Shape mismatch: y_true {yt.shape} vs y_pred {yp.shape}")

    err = yp - yt
    abs_err = np.abs(err)

    mae = float(np.mean(abs_err))
    rmse_val = float(np.sqrt(np.mean(err ** 2)))

    tw_ae = np.mean(abs_err, axis=1)  # time-wise mean AE
    nw_ae = np.mean(abs_err, axis=0)  # node-wise mean AE

    return {
        "MAE": mae,
        "RMSE": rmse_val,
        "Max_TW_AE": float(np.max(tw_ae)),
        "Max_NW_AE": float(np.max(nw_ae)),
    }


# ============================================================
# Save per-node time-series CSVs (same folder, similar naming)
# ============================================================
def save_node_timeseries_csv(
    out_dir: Path,
    node_id: int,
    phase: str,
    d_interval: int,
    node_actuals_full: np.ndarray,
    node_preds_full: np.ndarray,
    tag: str = "GCN",
):
    """
    Saves 2 CSVs (full + interval) in the SAME folder you specified,
    with naming style similar to your previous files, only suffix differs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    node_actuals_full = np.asarray(node_actuals_full).reshape(-1)
    node_preds_full = np.asarray(node_preds_full).reshape(-1)

    # FULL
    t_full = np.arange(len(node_actuals_full), dtype=int)
    df_full = pd.DataFrame({
        "t_index": t_full,
        "y_true": node_actuals_full,
        "y_pred": node_preds_full,
        "abs_err": np.abs(node_preds_full - node_actuals_full),
        "sq_err": (node_preds_full - node_actuals_full) ** 2,
    })
    out_full = out_dir / f"node_{node_id}_phase_{phase}_timeseries_full_{tag}_random.csv"
    df_full.to_csv(out_full, index=False)

    # INTERVAL
    idx = np.arange(0, len(node_actuals_full), int(d_interval), dtype=int)
    df_int = pd.DataFrame({
        "t_index": idx,
        "y_true": node_actuals_full[idx],
        "y_pred": node_preds_full[idx],
        "abs_err": np.abs(node_preds_full[idx] - node_actuals_full[idx]),
        "sq_err": (node_preds_full[idx] - node_actuals_full[idx]) ** 2,
    })
    out_int = out_dir / f"node_{node_id}_phase_{phase}_timeseries_interval_{int(d_interval)}_{tag}_random.csv"
    df_int.to_csv(out_int, index=False)

    return out_full, out_int


# ============================================================
# Vanilla GCN model (static graph, batched by node-offset trick)
# ============================================================
class StaticGCN(nn.Module):
    def __init__(
        self,
        in_feats,
        hidden_feats,
        out_feats,
        num_all_nodes,
        hrrsm_indices,
        sm_indices,
        num_gnn_layers=2,
        dropout=0.2,
    ):
        super().__init__()
        self.num_all_nodes = num_all_nodes
        self.hrrsm_indices = hrrsm_indices
        self.sm_indices = sm_indices
        self.hidden_feats = hidden_feats

        self.gnn_layers = nn.ModuleList()
        self.gnn_layers.append(GCNConv(in_feats, hidden_feats))
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GCNConv(hidden_feats, hidden_feats))

        self.dropout = nn.Dropout(dropout)

        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats // 2),
            nn.ReLU(),
            nn.Linear(hidden_feats // 2, out_feats),
        )

    def forward(self, x_input_batch, edge_index, edge_weight=None):
        device = x_input_batch.device
        B = x_input_batch.shape[0]
        N = self.num_all_nodes
        E = edge_index.size(1)

        x_full = torch.zeros(B, N, 1, device=device)
        x_full[:, self.hrrsm_indices, 0] = x_input_batch

        h = x_full.view(B * N, 1)

        if B == 1:
            edge_index_b = edge_index
            edge_weight_b = edge_weight
        else:
            edge_index_b = edge_index.repeat(1, B)
            offsets = torch.arange(B, device=device).repeat_interleave(E) * N
            edge_index_b = edge_index_b + offsets
            edge_weight_b = edge_weight.repeat(B) if edge_weight is not None else None

        for i, conv in enumerate(self.gnn_layers):
            h = conv(h, edge_index_b, edge_weight_b)
            h = F.relu(h)
            if i < len(self.gnn_layers) - 1:
                h = self.dropout(h)

        h_full = h.view(B, N, self.hidden_feats)
        h_sm = h_full[:, self.sm_indices, :]
        out = self.prediction_head(h_sm).squeeze(-1)  # [B, N_sm]
        return out


# ============================================================
# CONFIG
# ============================================================
PHASE = "A"
NODE_TO_PLOT_ID = 445
PLOT_START = 0
PLOT_END = 200
d_interval = 15
BATCH_SIZE = 32

# random.seed(42)
# HRRSM_NODES = sorted(random.sample(range(1, 460), 67))
# print("HRRSM_NODES (count = {}):".format(len(HRRSM_NODES)))
# print(HRRSM_NODES)

HRRSM_NODES = [1, 3, 8, 16, 20, 25, 28, 31, 44, 52, 57, 58, 66, 79, 85, 97, 106, 123, 136, 149, 169, 201, 204, 206, 212,
               231, 234, 237, 249, 253, 265, 269, 277, 280, 282, 283, 289, 290, 293, 299, 313, 315, 322, 327, 332, 333,
               339, 344, 346, 349, 351, 365, 371, 374, 376, 379, 383, 398, 405, 409, 425, 437, 443, 450, 451, 452, 455]

# model_load_path = f"best_model_static_gcn_random_pyG_Phase{PHASE}.pt"

model_load_path = f"best_model_static_gcnpyG_Phase{PHASE}.pt"
# ============================================================
# Load data
# ============================================================
print("--- Loading Data and Creating Graph ---")
loader = StaticVoltageLoader(data_dir=".", phase=PHASE, hrrsm_node_list=HRRSM_NODES)
train_loader, val_loader, test_loader, adj_matrix_t, mean_t, std_t = loader.get_dataloaders(
    batch_size=BATCH_SIZE, shuffle=True
)

adj_matrix_t = adj_matrix_t.to(device)
mean_t = mean_t.to(device)
std_t = std_t.to(device)

num_all_nodes = len(loader.all_nodes_list_str)
hrrsm_indices = torch.tensor(loader.hrrsm_indices).long().to(device)
sm_indices = torch.tensor(loader.sm_indices).long().to(device)

sm_means = mean_t[sm_indices]
sm_stds = std_t[sm_indices]

print("--- Creating PyG edge_index and edge_weight ---")
src_nodes, dst_nodes = adj_matrix_t.nonzero(as_tuple=True)
edge_index = torch.stack([src_nodes, dst_nodes], dim=0).long().to(device)
edge_weight = adj_matrix_t[src_nodes, dst_nodes].float().to(device)
print(f"  Graph tensors created with {num_all_nodes} nodes and {edge_index.size(1)} edges.")


# ============================================================
# Build model
# ============================================================
model = StaticGCN(
    in_feats=1,
    hidden_feats=256,
    out_feats=1,
    num_all_nodes=num_all_nodes,
    hrrsm_indices=hrrsm_indices,
    sm_indices=sm_indices,
    num_gnn_layers=3,
    dropout=0.2,
).to(device)


# ============================================================
# Load weights
# ============================================================
print("\n" + "=" * 70)
print(f"🚀 Starting Test Set Evaluation for **Vanilla GCN | PHASE {PHASE}**")
print("=" * 70)

try:
    state = torch.load(model_load_path, map_location=device, weights_only=True)
except TypeError:
    state = torch.load(model_load_path, map_location=device)

model.load_state_dict(state)
model.eval()
print(f"✅ Loaded weights: {model_load_path}")


# ============================================================
# TEST inference (RAW only)
# ============================================================
with torch.no_grad():
    all_predictions, all_actuals = [], []
    for x_input, y_target in test_loader:
        x_input = x_input.to(device)
        y_target = y_target.to(device)

        y_hat = model(x_input, edge_index, edge_weight)  # scaled
        all_predictions.append(y_hat)
        all_actuals.append(y_target)

predictions_t = torch.cat(all_predictions)  # [T_test, N_sm] scaled
actuals_t = torch.cat(all_actuals)          # [T_test, N_sm] scaled

# Unscale to PU
predictions_unscaled = (predictions_t * sm_stds) + sm_means
actuals_unscaled = (actuals_t * sm_stds) + sm_means

predictions_np = predictions_unscaled.detach().cpu().numpy()
actuals_np = actuals_unscaled.detach().cpu().numpy()


# ============================================================
# FULL TEST SCALARS (PRINT EXACTLY LIKE Evaluation_random)
# ============================================================
m_full = fullset_scalar_metrics(actuals_np, predictions_np)

print("\n=== FULL TEST (RAW, all timesteps) SCALARS ===")
print(f"MAE      : {m_full['MAE']:.6f}")
print(f"RMSE     : {m_full['RMSE']:.6f}")
print(f"Max TW-AE: {m_full['Max_TW_AE']:.6f}")
print(f"Max NW-AE: {m_full['Max_NW_AE']:.6f}")

T_test = actuals_np.shape[0]
interval_idx = np.arange(0, T_test, int(d_interval), dtype=int)

m_int = fullset_scalar_metrics(actuals_np[interval_idx, :], predictions_np[interval_idx, :])
print(f"\n=== FULL TEST (RAW, interval step = {int(d_interval)}) SCALARS ===")
print(f"MAE      : {m_int['MAE']:.6f}")
print(f"RMSE     : {m_int['RMSE']:.6f}")
print(f"Max TW-AE: {m_int['Max_TW_AE']:.6f}")
print(f"Max NW-AE: {m_int['Max_NW_AE']:.6f}")

# Save these scalars to CSV (in the SAME folder you specified)
scalars_csv = SAVED_TIMESERIES_DIR / f"FULL_TEST_SCALARS_RAW_GCN_Phase{PHASE}.csv"
pd.DataFrame([{
    "Phase": PHASE,
    "d_interval": int(d_interval),
    "MAE_full": m_full["MAE"],
    "RMSE_full": m_full["RMSE"],
    "Max_TW_AE_full": m_full["Max_TW_AE"],
    "Max_NW_AE_full": m_full["Max_NW_AE"],
    "MAE_int": m_int["MAE"],
    "RMSE_int": m_int["RMSE"],
    "Max_TW_AE_int": m_int["Max_TW_AE"],
    "Max_NW_AE_int": m_int["Max_NW_AE"],
}]).to_csv(scalars_csv, index=False)
print(f"✅ Saved scalars CSV: {scalars_csv}")


# ============================================================
# ALL-SM-NODES error table (whole test set)
# ============================================================
print("\n--- Computing ALL-SM-NODES error table (RAW GCN, whole test set) ---")

SM_NODE_IDS = np.array([int(loader.all_nodes_list_str[i]) for i in loader.sm_indices], dtype=int)

mae_full_vec, rmse_full_vec, maxae_full_vec, nrmse_full_vec, mape_full_vec, smape_full_vec = per_node_metrics(actuals_np, predictions_np)

act_int = actuals_np[interval_idx, :]
pred_int = predictions_np[interval_idx, :]
mae_int_vec, rmse_int_vec, maxae_int_vec, nrmse_int_vec, mape_int_vec, smape_int_vec = per_node_metrics(act_int, pred_int)

V_BASE_120 = 120.0
V_BASE_240 = 240.0

df_all = pd.DataFrame({
    "NodeID": SM_NODE_IDS,

    "MAE_PU_full": mae_full_vec,
    "RMSE_PU_full": rmse_full_vec,
    "MaxAE_PU_full": maxae_full_vec,
    "NRMSE_full": nrmse_full_vec,
    "MAPE_full_%": mape_full_vec,
    "SMAPE_full_%": smape_full_vec,

    f"MAE_PU_int_{int(d_interval)}": mae_int_vec,
    f"RMSE_PU_int_{int(d_interval)}": rmse_int_vec,
    f"MaxAE_PU_int_{int(d_interval)}": maxae_int_vec,
    f"NRMSE_int_{int(d_interval)}": nrmse_int_vec,
    f"MAPE_int_{int(d_interval)}_%": mape_int_vec,
    f"SMAPE_int_{int(d_interval)}_%": smape_int_vec,

    "MAE_120V_full": mae_full_vec * V_BASE_120,
    "RMSE_120V_full": rmse_full_vec * V_BASE_120,
    "MaxAE_120V_full": maxae_full_vec * V_BASE_120,
    f"MAE_120V_int_{int(d_interval)}": mae_int_vec * V_BASE_120,
    f"RMSE_120V_int_{int(d_interval)}": rmse_int_vec * V_BASE_120,
    f"MaxAE_120V_int_{int(d_interval)}": maxae_int_vec * V_BASE_120,

    "MAE_240V_full": mae_full_vec * V_BASE_240,
    "RMSE_240V_full": rmse_full_vec * V_BASE_240,
    "MaxAE_240V_full": maxae_full_vec * V_BASE_240,
    f"MAE_240V_int_{int(d_interval)}": mae_int_vec * V_BASE_240,
    f"RMSE_240V_int_{int(d_interval)}": rmse_int_vec * V_BASE_240,
    f"MaxAE_240V_int_{int(d_interval)}": maxae_int_vec * V_BASE_240,
})

df_all = df_all.sort_values("MaxAE_PU_full", ascending=False).reset_index(drop=True)

out_csv = SAVED_TIMESERIES_DIR / f"ALL_SM_NODES_ErrorTable_RAW_GCN_Phase{PHASE}.csv"
df_all.to_csv(out_csv, index=False)
print(f"✅ Saved ALL-NODES RAW error table: {out_csv}")

global_mae = float(np.mean(np.abs(actuals_np - predictions_np)))
global_rmse = float(np.sqrt(np.mean((actuals_np - predictions_np) ** 2)))
print(f"[GLOBAL over ALL times & ALL SM nodes] MAE={global_mae:.6f}, RMSE={global_rmse:.6f}")

print("\nTop-10 worst nodes by MaxAE_PU_full:")
print(df_all[["NodeID", "MaxAE_PU_full", "RMSE_PU_full", "MAE_PU_full"]].head(10).to_string(index=False))


# ============================================================
# Single-node plot + residual plot (RAW only)
# ============================================================
SM_NODE_ID_LIST = [int(loader.all_nodes_list_str[i]) for i in loader.sm_indices]
if NODE_TO_PLOT_ID not in SM_NODE_ID_LIST:
    print(f"\n❌ ERROR: Node {NODE_TO_PLOT_ID} is not a valid SM node.")
    print(f"Choose from: {SM_NODE_ID_LIST}")
    raise SystemExit(1)

node_plot_index = SM_NODE_ID_LIST.index(NODE_TO_PLOT_ID)
print(f"\nMapped Node {NODE_TO_PLOT_ID} -> SM column index {node_plot_index}")

node_actuals_full = actuals_np[:, node_plot_index]
node_preds_full = predictions_np[:, node_plot_index]
total_len = len(node_actuals_full)

mae_full_raw = mean_absolute_error(node_actuals_full, node_preds_full)
rmse_full_raw = np.sqrt(mean_squared_error(node_actuals_full, node_preds_full))
maxae_full_raw = max_ae(node_actuals_full, node_preds_full)

interval_indices_full = np.arange(0, total_len, int(d_interval))
flat_actuals_interval_full = node_actuals_full[interval_indices_full]
flat_preds_interval_full = node_preds_full[interval_indices_full]

mae_int_raw = mean_absolute_error(flat_actuals_interval_full, flat_preds_interval_full)
rmse_int_raw = np.sqrt(mean_squared_error(flat_actuals_interval_full, flat_preds_interval_full))
maxae_int_raw = max_ae(flat_actuals_interval_full, flat_preds_interval_full)

print(f"\n--- Node {NODE_TO_PLOT_ID} RAW TEST metrics (GCN) ---")
print(f"  FULL: MAE={mae_full_raw:.6f}, RMSE={rmse_full_raw:.6f}, MaxAE={maxae_full_raw:.6f}")
print(f"  INT : MAE={mae_int_raw:.6f}, RMSE={rmse_int_raw:.6f}, MaxAE={maxae_int_raw:.6f}")


# ============================================================
# Save per-node time-series CSVs (IN YOUR saved_timeseries FOLDER)
# ============================================================
out_full_node_csv, out_int_node_csv = save_node_timeseries_csv(
    out_dir=SAVED_TIMESERIES_DIR,
    node_id=NODE_TO_PLOT_ID,
    phase=PHASE,
    d_interval=d_interval,
    node_actuals_full=node_actuals_full,
    node_preds_full=node_preds_full,
    tag="GCN",
)
print(f"✅ Saved node timeseries CSV (full):     {out_full_node_csv}")
print(f"✅ Saved node timeseries CSV (interval): {out_int_node_csv}")


# Plot folder (unchanged)
PLOT_DIR = os.path.join("samples", "Figure")
os.makedirs(PLOT_DIR, exist_ok=True)

FIG_W, FIG_H = 3.33, 2.30
DPI = 300

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

ps = max(0, int(PLOT_START))
pe = total_len if PLOT_END is None else min(total_len, int(PLOT_END))
if pe <= ps:
    ps, pe = 0, min(total_len, 800)

node_actuals_plot = node_actuals_full[ps:pe]
node_preds_plot = node_preds_full[ps:pe]

interval_indices_plot = np.arange(ps, pe, int(d_interval))
flat_actuals_interval_plot = node_actuals_full[interval_indices_plot]
flat_preds_interval_plot = node_preds_full[interval_indices_plot]

# --- main plot
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
x_plot = np.arange(ps, pe)

ax.plot(x_plot, node_preds_plot, label="Predicted", color="red", linestyle="--", linewidth=1.0)
ax.plot(x_plot, node_actuals_plot, label="Actual", color="blue", linestyle="-", linewidth=1.2)

if interval_indices_plot.size > 0:
    ax.scatter(interval_indices_plot, flat_preds_interval_plot, color="orange", s=10,
               label=f"Predict @ {int(d_interval)} min", zorder=2)
    ax.scatter(interval_indices_plot, flat_actuals_interval_plot, color="green", s=10,
               label=f"Actual @ {int(d_interval)} min", zorder=2)

data_lo = float(min(node_actuals_plot.min(), node_preds_plot.min()))
data_hi = float(max(node_actuals_plot.max(), node_preds_plot.max()))
yrange = max(1e-9, data_hi - data_lo)
pad = max(0.0002, 0.05 * yrange)
ax.set_ylim(data_lo - pad, data_hi + pad)

ax.yaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
ax.xaxis.set_major_locator(MaxNLocator(6))

ax.set_xlabel("Time step (min)")
ax.set_ylabel("Voltage (p.u.)")

ax.legend(loc="best", frameon=True, framealpha=1.0, facecolor="white", edgecolor="0.3")
ax.grid(False)
fig.subplots_adjust(left=0.14, right=0.995, bottom=0.22, top=0.98)

base = os.path.join(PLOT_DIR, f"{NODE_TO_PLOT_ID}{PHASE}_GCN")
fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(base + ".png", dpi=DPI, bbox_inches="tight", pad_inches=0.01)
plt.show()
plt.close(fig)
print(f"Saved: {base}.pdf and {base}.png")

# --- residual plot
residuals = node_actuals_full - node_preds_full
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

ax.plot(x_plot, residuals[ps:pe], label=f"Residual (MAE {mae_full_raw:.4f})", color="purple", linewidth=1.2)
ax.axhline(0, color="black", linestyle="--", linewidth=0.9, label="Zero")

ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.set_xlim(max(0, ps - 0.03 * (pe - ps)), (pe - 1) + 0.03 * (pe - ps))

ax.set_xlabel("Time step (min)")
ax.set_ylabel("Residual")

ax.legend(loc="best", frameon=True, framealpha=1.0, facecolor="white", edgecolor="0.3")
ax.grid(False)
fig.subplots_adjust(left=0.14, right=0.995, bottom=0.22, top=0.98)

base = os.path.join(PLOT_DIR, f"{NODE_TO_PLOT_ID}{PHASE}_GCN_residual")
fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(base + ".png", dpi=DPI, bbox_inches="tight", pad_inches=0.01)
plt.show()
plt.close(fig)
print(f"Saved: {base}.pdf and {base}.png")

print("\n--- Vanilla GCN Evaluation Finished ---")
