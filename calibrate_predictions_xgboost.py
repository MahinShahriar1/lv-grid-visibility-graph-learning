import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
import pandas as pd
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.isotonic import IsotonicRegression
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import GradientBoostingRegressor
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

plt.rcParams.update({"font.family": ["Linux Libertine"]})
# --- NEW IMPORT FOR GBR SPLIT ---
try:
    from sklearn.model_selection import train_test_split
except ImportError:
    print("Warning: sklearn.model_selection not found. Calibrator Early Stopping will be disabled.")
    train_test_split = None

# --- NEW IMPORT FOR XGBOOST ---
try:
    from xgboost import XGBRegressor # type: ignore
except ImportError:
    print("=" * 50)
    print("❌ ERROR: Could not import 'xgboost'.")
    print("Please install it: pip install xgboost")
    print("=" * 50)
    XGBRegressor = None


# ---------- NEW METRIC HELPERS ----------
def max_ae(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.max(np.abs(y_true - y_pred)))


def nrmse(y_true, y_pred, eps=1e-8):
    """RMSE normalized by range of y_true."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    denom = float(y_true.max() - y_true.min())
    if denom < eps:
        return np.nan
    return float(rmse / denom)


def mape(y_true, y_pred, eps=1e-8):
    """Mean Absolute Percentage Error in %."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    denom = np.clip(np.abs(y_true), eps, None)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true, y_pred, eps=1e-8):
    """Symmetric MAPE in %."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    num = np.abs(y_true - y_pred)
    denom = np.clip((np.abs(y_true) + np.abs(y_pred)) / 2.0, eps, None)
    return float(np.mean(num / denom) * 100.0)


# --- 1. IMPORT THE CORRECT LOADER ---
try:
    from load_graphsage_data import StaticVoltageLoader  # type: ignore
except ImportError:
    print("=" * 50)
    print("ERROR: Could not find 'load_graphsage_data.py'")
    print("Please keep load_graphsage_data.py in the same directory.")
    print("=" * 50)
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Using device: {device} for Evaluation ---")


# =================================================================
# --- 2. MODEL ARCHITECTURE (MANDATORY FOR LOADING WEIGHTS) ---
# =================================================================
class WeightedSAGEConv(MessagePassing):
    """
    PyG-style weighted GraphSAGE.

    x: [N_total, in_feats]
    edge_index: [2, E_total]  (src, dst)
    edge_weight: [E_total] or [E_total, 1]
    """
    def __init__(self, in_feat, out_feat):
        super().__init__(aggr="add")  # sum messages (same as fn.sum)
        self.linear = nn.Linear(in_feat * 2, out_feat)

    def forward(self, x, edge_index, edge_weight):
        # x: [N, F]
        # edge_index: [2, E]
        if edge_weight.dim() == 1:
            edge_weight = edge_weight.view(-1, 1)  # [E, 1]

        # 1) aggregate weighted messages: h_sum_j = sum_i w_ij * h_i
        h_sum = self.propagate(edge_index=edge_index, x=x, edge_weight=edge_weight)  # [N, F]

        # 2) compute sum of |w| per destination node: w_abs_sum_j = sum_i |w_ij|
        src, dst = edge_index  # [E], [E]
        w_abs = edge_weight.abs()  # [E, 1]
        w_abs_sum = scatter(
            w_abs,
            dst,
            dim=0,
            dim_size=x.size(0),
            reduce="sum"
        )  # [N, 1]

        denom = w_abs_sum.clamp_min(1e-6)  # avoid division by zero

        # 3) neighbor aggregate: h_N_j = h_sum_j / w_abs_sum_j
        h_N = h_sum / denom  # [N, F]

        # 4) concat self + neighbors and apply linear (same as your DGL code)
        out = self.linear(torch.cat([x, h_N], dim=1))
        return out

    def message(self, x_j, edge_weight):
        """
        x_j: source node features [E, F]
        edge_weight: [E, 1]
        m_ij = w_ij * h_i   (same as u_mul_e("h", "w"))
        """
        return x_j * edge_weight


class StaticGraphSAGE(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats,
                 num_all_nodes, hrrsm_indices, sm_indices,
                 num_gnn_layers=2, dropout=0.2):
        super().__init__()
        self.num_all_nodes = num_all_nodes
        self.hrrsm_indices = hrrsm_indices
        self.sm_indices = sm_indices
        self.hidden_feats = hidden_feats

        self.gnn_layers = nn.ModuleList([WeightedSAGEConv(in_feats, hidden_feats)])
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(WeightedSAGEConv(hidden_feats, hidden_feats))

        self.dropout = nn.Dropout(dropout)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats // 2),
            nn.ReLU(),
            nn.Linear(hidden_feats // 2, out_feats)
        )

    def forward(self, x_input_batch, edge_index, edge_weight):
        """
        x_input_batch: [B, num_HRRSM_nodes]
        edge_index: [2, E] for the single feeder graph (no batch)
        edge_weight: [E]
        """
        device = x_input_batch.device
        batch_size = x_input_batch.shape[0]

        # 1) Put HRRSM measurements into full node feature tensor
        x_full = torch.zeros(batch_size, self.num_all_nodes, 1, device=device)
        x_full[:, self.hrrsm_indices, 0] = x_input_batch  # like before

        # 2) Flatten to [B * N, F]
        h = x_full.view(batch_size * self.num_all_nodes, 1)

        # 3) Build batched edge_index/edge_weight by offsetting node IDs
        N = self.num_all_nodes
        E = edge_index.size(1)

        if batch_size == 1:
            edge_index_batch = edge_index
            edge_weight_batch = edge_weight
        else:
            # repeat the edges for each element in the batch
            edge_index_batch = edge_index.repeat(1, batch_size)  # [2, B*E]
            offsets = torch.arange(batch_size, device=device).repeat_interleave(E) * N  # [B*E]
            edge_index_batch = edge_index_batch + offsets  # broadcast to both rows
            edge_weight_batch = edge_weight.repeat(batch_size)  # [B*E]

        # 4) GNN layers
        for i, layer in enumerate(self.gnn_layers):
            h = layer(h, edge_index_batch, edge_weight_batch)
            h = F.relu(h)
            if i < len(self.gnn_layers) - 1:
                h = self.dropout(h)

        # 5) Reshape back to [B, N, hidden_feats]
        h_full_nodes = h.view(batch_size, self.num_all_nodes, self.hidden_feats)

        # 6) Select SM nodes and predict
        h_target_nodes = h_full_nodes[:, self.sm_indices, :]  # [B, Ns, hidden_feats]
        out = self.prediction_head(h_target_nodes)            # [B, Ns, 1]
        return out.squeeze(-1)                                # [B, Ns]


# =================================================================
# --- 3. CONFIGURATION (SET THESE) ---
# =================================================================
PHASE = "B"
NODE_TO_PLOT_ID =  81
PLOT_START =60
PLOT_END = 160
d_interval = 15

APPLY_CALIBRATION = False  # Master switch: Do we even *consider* calibration?

# --- THIS IS YOUR "HIGH ERROR" TRIGGER ---
# Set to 0.0 to FORCE calibration to run every time
CALIBRATION_MAX_AE_THRESHOLD = 0.0003

HRRSM_NODES = [1, 3, 8, 16, 20, 25, 28, 31, 44, 52, 57, 58, 66, 79, 85, 97, 106, 123, 136, 149, 169, 201, 204, 206, 212,
               231, 234, 237, 249, 253, 265, 269, 277, 280, 282, 283, 289, 290, 293, 299, 313, 315, 322, 327, 332, 333,
               339, 344, 346, 349, 351, 365, 371, 374, 376, 379, 383, 398, 405, 409, 425, 437, 443, 450, 451, 452, 455]
# HRRSM_NODES = [
#     4, 13, 14, 16, 17, 23, 41, 45, 48, 50, 53, 58, 64, 72, 80, 82,
#     102, 111, 112, 113, 115, 120, 126, 136, 141, 143, 151, 173,
#     175, 177, 184, 194, 195, 215, 217, 230, 236, 259, 275, 280,
#     283, 288, 302, 303, 309, 310, 317, 322, 328, 333, 347, 358,
#     360, 367, 374, 378, 380, 389, 391, 413, 414, 415, 425, 434,
#     446, 454, 457
# ]

BATCH_SIZE = 32

model_load_path = f"best_model_static_graphsage5pyG_PhaseA.pt"

# =================================================================
# --- 4. DATA LOADING ---
# =================================================================
print("--- Loading Data and Creating Graph ---")

loader = StaticVoltageLoader(
    data_dir=".",
    phase=PHASE,
    hrrsm_node_list=HRRSM_NODES,
)

train_loader, val_loader, test_loader, adj_matrix_t, mean_t, std_t = loader.get_dataloaders(
    batch_size=BATCH_SIZE,
    shuffle=False,
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
src_nodes, dst_nodes = adj_matrix_t.nonzero(as_tuple=True)  # both [E]
edge_index = torch.stack([src_nodes, dst_nodes], dim=0).long().to(device)  # [2, E]
edge_weight = adj_matrix_t[src_nodes, dst_nodes].float().to(device)        # [E]
print(f"  Graph tensors created with {num_all_nodes} nodes and {edge_index.size(1)} edges.")

# Instantiate the model (must match training config!)
model = StaticGraphSAGE(
    in_feats=1,
    hidden_feats=256,  # same as training
    out_feats=1,
    num_all_nodes=num_all_nodes,
    hrrsm_indices=hrrsm_indices,
    sm_indices=sm_indices,
    num_gnn_layers=3,
    dropout=0.2,
).to(device)

# =================================================================
# --- 5. LOAD WEIGHTS ---
# =================================================================
print("\n" + "=" * 70)
print(f"🚀 Starting Test Set Evaluation for **PHASE {PHASE}**")
print("=" * 70)

try:
    state = torch.load(model_load_path, map_location=device, weights_only=True)
except TypeError:
    state = torch.load(model_load_path, map_location=device)

model.load_state_dict(state)
model.eval()

# =================================================================
# --- 5.1 COLLECT VALIDATION PREDICTIONS (INCLUDES HRRSM INPUTS) ---
# =================================================================
with torch.no_grad():
    val_x_inputs, val_preds, val_tgts = [], [], []  # --- ADDED val_x_inputs
    for x_input, y_target in val_loader:
        x_input_gpu = x_input.to(device)
        y_target = y_target.to(device)
        y_hat = model(x_input_gpu, edge_index, edge_weight)  # [B, N_sm], scaled

        val_x_inputs.append(x_input.cpu())  # --- ADDED
        val_preds.append(y_hat)
        val_tgts.append(y_target)

if len(val_preds) > 0:
    val_x_input_t = torch.cat(val_x_inputs)          # CPU
    val_pred_t    = torch.cat(val_preds)             # GPU
    val_act_t     = torch.cat(val_tgts)              # GPU

    # Unscale (still on GPU)
    val_pred_unscaled = (val_pred_t * sm_stds) + sm_means
    val_act_unscaled  = (val_act_t  * sm_stds) + sm_means

    # Now move to CPU + NumPy
    val_x_input_np = val_x_input_t.numpy()                               # [T_val, N_hrrsm]
    val_pred_np    = val_pred_unscaled.detach().cpu().numpy()
    val_act_np     = val_act_unscaled.detach().cpu().numpy()
else:
    print("[Warning] Validation loader empty; node-wise calibration will be disabled.")
    APPLY_CALIBRATION = False
    val_pred_np = None
    val_act_np = None
    val_x_input_np = None  # --- ADDED

# =================================================================
# --- 6. TESTING & UN-SCALING (INCLUDES HRRSM INPUTS) ---
# =================================================================
with torch.no_grad():
    all_x_inputs, all_predictions, all_actuals = [], [], []
    for x_input, y_target in test_loader:
        x_input_gpu  = x_input.to(device)
        y_target_gpu = y_target.to(device)
        y_hat        = model(x_input_gpu, edge_index, edge_weight)

        # HRRSM inputs: CPU (for XGBoost)
        all_x_inputs.append(x_input.cpu())

        # predictions & targets: GPU
        all_predictions.append(y_hat)
        all_actuals.append(y_target_gpu)

test_x_input_t = torch.cat(all_x_inputs)        # CPU
predictions_t  = torch.cat(all_predictions)     # GPU
actuals_t      = torch.cat(all_actuals)         # GPU

# Unscale on GPU
predictions_unscaled = (predictions_t * sm_stds) + sm_means
actuals_unscaled     = (actuals_t    * sm_stds) + sm_means

# Then to NumPy
test_x_input_np = test_x_input_t.numpy()
predictions_np  = predictions_unscaled.detach().cpu().numpy()
actuals_np      = actuals_unscaled.detach().cpu().numpy()

# Keep raw copy (uncalibrated)
predictions_np_raw = predictions_np.copy()
def fullset_scalar_metrics(actuals_2d: np.ndarray, preds_2d: np.ndarray) -> dict:
    err = preds_2d - actuals_2d
    ae  = np.abs(err)

    mae = float(np.mean(ae))
    rmse = float(np.sqrt(mean_squared_error(actuals_2d.reshape(-1), preds_2d.reshape(-1))))

    tw_ae = np.mean(ae, axis=1)          # [T]
    max_tw_ae = float(np.max(tw_ae))     # scalar

    nw_ae = np.mean(ae, axis=0)          # [N]
    max_nw_ae = float(np.max(nw_ae))     # scalar

    return {"MAE": mae, "RMSE": rmse, "Max_TW_AE": max_tw_ae, "Max_NW_AE": max_nw_ae}

# --- normal (all timesteps) ---
m_full = fullset_scalar_metrics(actuals_np, predictions_np_raw)
print("\n=== FULL TEST (RAW, all timesteps) SCALARS ===")
print(f"MAE      : {m_full['MAE']:.6f}")
print(f"RMSE     : {m_full['RMSE']:.6f}")
print(f"Max TW-AE: {m_full['Max_TW_AE']:.6f}")
print(f"Max NW-AE: {m_full['Max_NW_AE']:.6f}")

# --- interval (every d_interval samples) ---
T_test = actuals_np.shape[0]
interval_idx = np.arange(0, T_test, int(d_interval), dtype=int)

m_int = fullset_scalar_metrics(actuals_np[interval_idx, :], predictions_np_raw[interval_idx, :])
print(f"\n=== FULL TEST (RAW, interval step = {int(d_interval)}) SCALARS ===")
print(f"MAE      : {m_int['MAE']:.6f}")
print(f"RMSE     : {m_int['RMSE']:.6f}")
print(f"Max TW-AE: {m_int['Max_TW_AE']:.6f}")
print(f"Max NW-AE: {m_int['Max_NW_AE']:.6f}")
# =================================================================
# --- ALL-SM-NODES (WHOLE TEST SET) ERROR TABLE (RAW GNN) ---
#     Uses: actuals_np, predictions_np_raw  shape = [T_test, N_sm]
# =================================================================
print("\n--- Computing ALL-SM-NODES error table (RAW GNN, whole test set) ---")

# SM node IDs in same column order as actuals_np / predictions_np_raw
SM_NODE_IDS = np.array([int(loader.all_nodes_list_str[i]) for i in loader.sm_indices], dtype=int)

T_test, N_sm = actuals_np.shape
assert predictions_np_raw.shape == actuals_np.shape, "Shape mismatch: predictions vs actuals"

# Interval indices (every d_interval samples)
interval_idx = np.arange(0, T_test, d_interval, dtype=int)

def per_node_metrics(y_true_2d, y_pred_2d):
    """Column-wise metrics for 2D arrays [T, N]. Returns vectors of length N."""
    err = y_true_2d - y_pred_2d
    abs_err = np.abs(err)

    mae = abs_err.mean(axis=0)
    rmse = np.sqrt((err ** 2).mean(axis=0))
    maxae = abs_err.max(axis=0)

    y_range = (y_true_2d.max(axis=0) - y_true_2d.min(axis=0))
    nrmse_vec = np.where(y_range > 1e-8, rmse / y_range, np.nan)

    denom = np.clip(np.abs(y_true_2d), 1e-8, None)
    mape_vec = (np.abs(err) / denom).mean(axis=0) * 100.0

    denom2 = np.clip((np.abs(y_true_2d) + np.abs(y_pred_2d)) / 2.0, 1e-8, None)
    smape_vec = (abs_err / denom2).mean(axis=0) * 100.0

    return mae, rmse, maxae, nrmse_vec, mape_vec, smape_vec

# --- Full test set per-node metrics (PU) ---
mae_full, rmse_full, maxae_full, nrmse_full, mape_full, smape_full = per_node_metrics(
    actuals_np, predictions_np_raw
)

# --- Interval per-node metrics (PU) ---
act_int  = actuals_np[interval_idx, :]
pred_int = predictions_np_raw[interval_idx, :]
mae_int, rmse_int, maxae_int, nrmse_int, mape_int, smape_int = per_node_metrics(
    act_int, pred_int
)

# --- Convert PU absolute errors to volts ---
V_BASE_120 = 120.0
V_BASE_240 = 240.0

df_all = pd.DataFrame({ # type: ignore
    "NodeID": SM_NODE_IDS,

    # PU (full)
    "MAE_PU_full": mae_full,
    "RMSE_PU_full": rmse_full,
    "MaxAE_PU_full": maxae_full,
    "NRMSE_full": nrmse_full,
    "MAPE_full_%": mape_full,
    "SMAPE_full_%": smape_full,

    # PU (interval)
    f"MAE_PU_int_{d_interval}": mae_int,
    f"RMSE_PU_int_{d_interval}": rmse_int,
    f"MaxAE_PU_int_{d_interval}": maxae_int,
    f"NRMSE_int_{d_interval}": nrmse_int,
    f"MAPE_int_{d_interval}_%": mape_int,
    f"SMAPE_int_{d_interval}_%": smape_int,

    # 120V
    "MAE_120V_full": mae_full * V_BASE_120,
    "RMSE_120V_full": rmse_full * V_BASE_120,
    "MaxAE_120V_full": maxae_full * V_BASE_120,
    f"MAE_120V_int_{d_interval}": mae_int * V_BASE_120,
    f"RMSE_120V_int_{d_interval}": rmse_int * V_BASE_120,
    f"MaxAE_120V_int_{d_interval}": maxae_int * V_BASE_120,

    # 240V
    "MAE_240V_full": mae_full * V_BASE_240,
    "RMSE_240V_full": rmse_full * V_BASE_240,
    "MaxAE_240V_full": maxae_full * V_BASE_240,
    f"MAE_240V_int_{d_interval}": mae_int * V_BASE_240,
    f"RMSE_240V_int_{d_interval}": rmse_int * V_BASE_240,
    f"MaxAE_240V_int_{d_interval}": maxae_int * V_BASE_240,
})

# Sort worst-first
df_all = df_all.sort_values("MaxAE_PU_full", ascending=False).reset_index(drop=True)

out_csv = f"ALL_SM_NODES_ErrorTable_RAW_Phase{PHASE}.csv"
df_all.to_csv(out_csv, index=False)
print(f"✅ Saved ALL-NODES RAW error table: {out_csv}")

# --- Optional: ONE scalar for the whole test set (all times + all SM nodes) ---
global_mae  = float(np.mean(np.abs(actuals_np - predictions_np_raw)))
global_rmse = float(np.sqrt(np.mean((actuals_np - predictions_np_raw) ** 2)))
print(f"[GLOBAL over ALL times & ALL SM nodes] MAE={global_mae:.6f}, RMSE={global_rmse:.6f}")

print("\nTop-10 worst nodes by MaxAE_PU_full:")
print(df_all[["NodeID", "MaxAE_PU_full", "RMSE_PU_full", "MAE_PU_full"]].head(10).to_string(index=False))

# =================================================================
# --- 7. PER-NODE METRICS & CONDITIONAL CALIBRATION ---
# =================================================================
print("\n------ Visualization Setup ------")

SM_NODE_IDS = [int(loader.all_nodes_list_str[i]) for i in loader.sm_indices]
node_plot_index = -1
try:
    node_plot_index = SM_NODE_IDS.index(NODE_TO_PLOT_ID)
    print(f"Successfully mapped Node {NODE_TO_PLOT_ID} to column index {node_plot_index}.")
except ValueError:
    print(f"\n❌ ERROR: Node {NODE_TO_PLOT_ID} is not a valid SM node.")
    print(f"Choose from: {SM_NODE_IDS}")
    exit()  # Exit if node not found

if node_plot_index != -1:

    # ---- Extract this node's full test series (RAW) ----
    node_actuals_full = actuals_np[:, node_plot_index]
    node_predictions_full_raw = predictions_np_raw[:, node_plot_index]
    total_len = len(node_actuals_full)

    # ---- Calculate RAW TEST metrics (for final comparison) ----
    mae_full_raw = mean_absolute_error(node_actuals_full, node_predictions_full_raw)
    rmse_full_raw = np.sqrt(mean_squared_error(node_actuals_full, node_predictions_full_raw))
    maxae_full_raw = max_ae(node_actuals_full, node_predictions_full_raw)

    # --- FIX: ADD MISSING RAW RELATIVE METRIC CALCULATIONS HERE ---
    nrmse_full_raw = nrmse(node_actuals_full, node_predictions_full_raw)
    mape_full_raw = mape(node_actuals_full, node_predictions_full_raw)
    smape_full_raw = smape(node_actuals_full, node_predictions_full_raw)

    interval_indices_full = np.arange(0, total_len, d_interval)
    flat_actuals_interval_full = node_actuals_full[interval_indices_full]
    flat_preds_interval_full_raw = node_predictions_full_raw[interval_indices_full]
    mae_int_raw = mean_absolute_error(flat_actuals_interval_full, flat_preds_interval_full_raw)
    rmse_int_raw = np.sqrt(mean_squared_error(flat_actuals_interval_full, flat_preds_interval_full_raw))
    maxae_int_raw = max_ae(flat_actuals_interval_full, flat_preds_interval_full_raw)

    # --- ADD MISSING RAW INTERVAL RELATIVE METRIC CALCULATIONS HERE ---
    nrmse_int_raw = nrmse(flat_actuals_interval_full, flat_preds_interval_full_raw)
    mape_int_raw = mape(flat_actuals_interval_full, flat_preds_interval_full_raw)
    smape_int_raw = smape(flat_actuals_interval_full, flat_preds_interval_full_raw)
    # ------------------------------------------------------------------

    print(f"\n--- Node {NODE_TO_PLOT_ID} Metrics (Entire Test Set, RAW) ---")
    print(f"  MAE:    {mae_full_raw:.6f}, RMSE: {rmse_full_raw:.6f}, MaxAE: {maxae_full_raw:.6f}")
    print(f"--- Node {NODE_TO_PLOT_ID} Metrics (Interval {d_interval} steps, RAW TEST) ---")
    print(f"  MAE:    {mae_int_raw:.6f}, RMSE: {rmse_int_raw:.6f}")

    # =================================================================
    # --- 7.1 CALIBRATION TRIGGER & TRAINING (NOW ON HRRSM + GNN_PRED) ---
    # =================================================================

    calibrator_node = None  # Use a generic name
    node_predictions_full = node_predictions_full_raw.copy()  # Start with raw
    calibration_was_run = False
    calibrator_name = "GBR"  # Default

    # --- THIS IS THE NEW LOGIC (using Validation Data) ---
    if APPLY_CALIBRATION and (val_pred_np is not None) and (train_test_split is not None) and (
            XGBRegressor is not None):

        # --- 1. CHECK TRIGGER on Validation Set ---
        val_pred_node_check = val_pred_np[:, node_plot_index]
        val_act_node_check = val_act_np[:, node_plot_index]

        # --- Use WHOLE validation set for trigger ---
        max_ae_full_val = max_ae(val_act_node_check, val_pred_node_check)
        print(f"\n[Calib Check] Raw FULL MaxAE on VALIDATION set: {max_ae_full_val:.6f}")

        # --- 2. RUN CALIBRATION if trigger is met ---
        if max_ae_full_val > CALIBRATION_MAX_AE_THRESHOLD:
            print(
                f"  [!] Triggering calibration: Val MaxAE ({max_ae_full_val:.6f}) > Threshold ({CALIBRATION_MAX_AE_THRESHOLD:.6f})")

            # Set this flag initially to true, it will be set to False if the sanity check fails
            calibration_was_run = True

            # --- A. PREPARE DATA ---
            calibrator_name = "XGBoost"
            X_val_features = val_x_input_np
            X_val_gnn_pred = val_pred_np[:, node_plot_index].reshape(-1, 1)

            # Combine HRRSM inputs (Scaled) and GNN prediction (Unscaled)
            X_val_full_input = np.hstack([X_val_features, X_val_gnn_pred])

            # Target: Residual Error (Actual - Predicted) in VOLTS
            y_val_full_actual = val_act_np[:, node_plot_index]
            r_val_full_residual = y_val_full_actual - X_val_gnn_pred[:, 0]

            print(f"[{calibrator_name}] Training calibrator using all {X_val_full_input.shape[0]} validation samples.")

            # Split 80/20 for internal training/early stopping
            X_xgb_train, X_xgb_eval, r_xgb_train, r_xgb_eval = train_test_split(
                X_val_full_input,
                r_val_full_residual,
                test_size=0.2,
                shuffle=False,  # <--- THIS IS THE ONLY CHANGE NEEDED
                # random_state=42  # <--- Recommended for reproducibility
            )

            # --- B. TRAIN XGBOOST ---
            calibrator_node = XGBRegressor(
                n_estimators=50000,
                max_depth=1000,
                learning_rate=0.01,
                objective='reg:pseudohubererror',
                early_stopping_rounds=30,
                eval_metric="mae",
                n_jobs=-1,
                # --- NEW REGULARIZATION PARAMETERS ---
                # min_child_weight=1,  # Prevents splitting on very few noisy samples
                # reg_alpha=0.005,  # L1 Regularization (Feature Sparsity)
                # reg_lambda=0.005  # L2 Regularization (Weight Smoothing)
            )

            print(f"[{calibrator_name}] Training calibrator with early stopping...")

            calibrator_node.fit(
                X_xgb_train, r_xgb_train,
                eval_set=[(X_xgb_train, r_xgb_train), (X_xgb_eval, r_xgb_eval)],
                verbose=False
            )

            print(f"[{calibrator_name}] Convergence! XGB Best Iteration: {calibrator_node.best_iteration}")

            # --- C. PLOT CONVERGENCE ---
            print(f"[{calibrator_name}] Plotting calibrator convergence...")
            try:
                results = calibrator_node.evals_result()

                train_loss = results['validation_0']['mae']
                val_loss = results['validation_1']['mae']
                y_label = 'Mean Absolute Error (Volts)'

                epochs = len(train_loss)
                x_axis = range(0, epochs)

                plt.figure(figsize=(10, 6))
                plt.plot(x_axis, train_loss, label='Train', color='blue')
                plt.plot(x_axis, val_loss, label='Validation', color='orange')

                best_iter = calibrator_node.best_iteration
                if best_iter < len(val_loss):
                    plt.axvline(x=best_iter, color='green', linestyle='--', label=f'Best Iteration ({best_iter})')

                plt.xlabel("Iteration (Tree)")
                plt.ylabel(y_label)
                plt.title(f"{calibrator_name} Convergence (Node {NODE_TO_PLOT_ID})")
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(f"static_model_node{NODE_TO_PLOT_ID}_xgb_convergence.png", dpi=300)
                plt.show()

            except Exception as e:
                print(f"[Warning] Could not plot convergence: {e}")

            # =========================================================
            # 🚨 D. THE "DO NO HARM" VALIDATION SANITY CHECK 🚨
            # =========================================================

            # 1. Predict correction for the WHOLE validation set
            r_hat_val_full = calibrator_node.predict(X_val_full_input)

            # 2. Calculate Final Calibrated Validation Prediction
            val_pred_calibrated = X_val_gnn_pred[:, 0] + r_hat_val_full

            # 3. Calculate MAE for both RAW and CALIBRATED
            mae_val_raw = mean_absolute_error(y_val_full_actual, X_val_gnn_pred[:, 0])
            mae_val_cal = mean_absolute_error(y_val_full_actual, val_pred_calibrated)

            print(f"\n[Sanity Check] Validation MAE Comparison:")
            print(f"  Raw GNN MAE:      {mae_val_raw:.6f}")
            print(f"  Calibrated MAE:   {mae_val_cal:.6f}")

            # 4. Decide: Only keep calibrator if it REDUCES error
            if mae_val_cal < mae_val_raw:
                improvement_percent = (mae_val_raw - mae_val_cal) * 100 / mae_val_raw
                print(f"✅ PASSED: Calibrator improved MAE by {improvement_percent:.2f}%. Retaining model.")
                # calibration_was_run remains True, and calibrator_node is available
            else:
                print(f"❌ FAILED: Calibrator did not improve validation MAE. Discarding model.")
                calibration_was_run = False
                calibrator_node = None  # Explicitly discard the trained model

        else:
            print(
                f"  [Calibration] Skipped: Val MaxAE ({max_ae_full_val:.6f}) <= Threshold ({CALIBRATION_MAX_AE_THRESHOLD:.6f})")
            calibration_was_run = False
            calibrator_node = None

    elif not APPLY_CALIBRATION:
        print("\n[Calibration] Disabled (APPLY_CALIBRATION=False).")
    elif val_pred_np is None:
        print("\n[Calibration] Disabled (validation data empty).")
    elif train_test_split is None:
        print("\n[Calibration] Disabled (train_test_split not found).")
    elif XGBRegressor is None:
        print("\n[Calibration] Disabled (XGBoost not found).")

    # --- 3. APPLY CALIBRATOR (if it was built) to TEST set ---
    if calibration_was_run and calibrator_node is not None:
        print(f"\n--- Applying converged {calibrator_name} to TEST set ---")

        # Feature 1: ALL HRRSM inputs from test
        X_test_features = test_x_input_np
        # Feature 2: The GNN's raw prediction for our node
        X_test_gnn_pred = node_predictions_full_raw.reshape(-1, 1)

        # Combine them: [N_samples, 65] + [N_samples, 1] -> [N_samples, 66]
        X_test_full_input = np.hstack([X_test_features, X_test_gnn_pred])

        # The XGB output is the *correction*
        r_hat_test = calibrator_node.predict(X_test_full_input)

        # Add the correction to the raw prediction
        node_predictions_full = node_predictions_full_raw + r_hat_test

        delta_node = node_predictions_full - node_predictions_full_raw
        print("Node calibration delta stats (correction amount):")
        print(f"  mean |delta| = {np.mean(np.abs(delta_node)):.6e}")
        print(f"  max  |delta| = {np.max(np.abs(delta_node)):.6e}")

    # =================================================================
    # --- 7.2 PRINT FINAL METRICS (XGBoost vs RAW GNN) ---
    # =================================================================

    mae_full_cal = mean_absolute_error(node_actuals_full, node_predictions_full)
    rmse_full_cal = np.sqrt(mean_squared_error(node_actuals_full, node_predictions_full))
    maxae_full_cal = max_ae(node_actuals_full, node_predictions_full)
    nrmse_full_cal = nrmse(node_actuals_full, node_predictions_full)
    mape_full_cal = mape(node_actuals_full, node_predictions_full)
    smape_full_cal = smape(node_actuals_full, node_predictions_full)

    flat_preds_interval_full_cal = node_predictions_full[interval_indices_full]
    mae_int_cal = mean_absolute_error(flat_actuals_interval_full, flat_preds_interval_full_cal)
    rmse_int_cal = np.sqrt(mean_squared_error(flat_actuals_interval_full, flat_preds_interval_full_cal))
    maxae_int_cal = max_ae(flat_actuals_interval_full, flat_preds_interval_full_cal)
    nrmse_int_cal = nrmse(flat_actuals_interval_full, flat_preds_interval_full_cal)
    mape_int_cal = mape(flat_actuals_interval_full, flat_preds_interval_full_cal)
    smape_int_cal = smape(flat_actuals_interval_full, flat_preds_interval_full_cal)

    print(f"\n--- Node {NODE_TO_PLOT_ID} Metrics (Entire Test Set, FINAL) ---")
    print(f"  MAE:    {mae_full_cal:.6f} (raw {mae_full_raw:.6f})")
    print(f"  RMSE:   {rmse_full_cal:.6f} (raw {rmse_full_raw:.6f})")
    print(f"  MaxAE:  {maxae_full_cal:.6f} (raw {maxae_full_raw:.6f})")
    print(f"  NRMSE:  {nrmse_full_cal:.6f}")
    print(f"  MAPE%:  {mape_full_cal:.4f}")
    print(f"  SMAPE%: {smape_full_cal:.4f}")

    print(f"\n--- Node {NODE_TO_PLOT_ID} Metrics (Interval {d_interval} steps, FINAL TEST) ---")
    print(f"  MAE:    {mae_int_cal:.6f} (raw {mae_int_raw:.6f})")
    print(f"  RMSE:   {rmse_int_cal:.6f} (raw {rmse_int_raw:.6f})")
    print(f"  MaxAE:  {maxae_int_cal:.6f}")
    print(f"  NRMSE:  {nrmse_int_cal:.6f}")
    print(f"  MAPE%:  {mape_int_cal:.4f}")
    print(f"  SMAPE%: {smape_int_cal:.4f}")

    # =================================================================
    # --- 7.3 ACTUAL VOLTAGE ERROR CALCULATION (for Matrix) ---
    # =================================================================

    # --- Defined LV Base Voltages for Conversion (V_base) ---
    V_BASE_120 = 120.0  # Phase-to-Neutral
    V_BASE_240 = 240.0  # Phase-to-Phase

    # Calculate ALL Absolute Errors in Volts

    # 120V Base Conversions
    mae_raw_v120 = mae_full_raw * V_BASE_120
    rmse_raw_v120 = rmse_full_raw * V_BASE_120
    maxae_raw_v120 = maxae_full_raw * V_BASE_120
    mae_int_raw_v120 = mae_int_raw * V_BASE_120
    rmse_int_raw_v120 = rmse_int_raw * V_BASE_120

    mae_cal_v120 = mae_full_cal * V_BASE_120
    rmse_cal_v120 = rmse_full_cal * V_BASE_120
    maxae_cal_v120 = maxae_full_cal * V_BASE_120
    mae_int_cal_v120 = mae_int_cal * V_BASE_120
    rmse_int_cal_v120 = rmse_int_cal * V_BASE_120

    # 240V Base Conversions
    mae_raw_v240 = mae_full_raw * V_BASE_240
    rmse_raw_v240 = rmse_full_raw * V_BASE_240
    maxae_raw_v240 = maxae_full_raw * V_BASE_240
    mae_int_raw_v240 = mae_int_raw * V_BASE_240
    rmse_int_raw_v240 = rmse_int_raw * V_BASE_240

    mae_cal_v240 = mae_full_cal * V_BASE_240
    rmse_cal_v240 = rmse_full_cal * V_BASE_240
    maxae_cal_v240 = maxae_full_cal * V_BASE_240
    mae_int_cal_v240 = mae_int_cal * V_BASE_240
    rmse_int_cal_v240 = rmse_int_cal * V_BASE_240

    # =================================================================
    # --- 7.4 FINAL COMPREHENSIVE ERROR MATRIX 📊 ---
    # =================================================================
    print("\n" + "=" * 140)
    title_cal_name = calibrator_name if (calibration_was_run and calibrator_node is not None) else "None"
    print(f"📊 FINAL ERROR MATRIX FOR NODE {NODE_TO_PLOT_ID} (Phase {PHASE}) - Calibrator: {title_cal_name}")
    print("=" * 140)

    # Note: Relative metrics (NRMSE, MAPE, SMAPE) are scale-invariant.
    data = [
        # Metric | Raw Full (PU) | Raw Int (PU) | Cal Full (PU) | Cal Int (PU) | Raw Full (120V) | Raw Int (120V) | Cal Full (120V) | Cal Int (120V) | Raw Full (240V) | Raw Int (240V) | Cal Full (240V) | Cal Int (240V)
        ["MAE", mae_full_raw, mae_int_raw, mae_full_cal, mae_int_cal, mae_raw_v120, mae_int_raw_v120, mae_cal_v120,
         mae_int_cal_v120, mae_raw_v240, mae_int_raw_v240, mae_cal_v240, mae_int_cal_v240],
        ["RMSE", rmse_full_raw, rmse_int_raw, rmse_full_cal, rmse_int_cal, rmse_raw_v120, rmse_int_raw_v120,
         rmse_cal_v120, rmse_int_cal_v120, rmse_raw_v240, rmse_int_raw_v240, rmse_cal_v240, rmse_int_cal_v240],
        ["MaxAE", maxae_full_raw, maxae_int_raw, maxae_full_cal, maxae_int_cal, maxae_raw_v120,
         maxae_int_raw * V_BASE_120, maxae_cal_v120, maxae_int_cal * V_BASE_120, maxae_raw_v240,
         maxae_int_raw * V_BASE_240, maxae_cal_v240, maxae_int_cal * V_BASE_240],
        ["NRMSE", nrmse_full_raw, nrmse_int_raw, nrmse_full_cal, nrmse_int_cal, nrmse_full_raw, nrmse_int_raw,
         nrmse_full_cal, nrmse_int_cal, nrmse_full_raw, nrmse_int_raw, nrmse_full_cal, nrmse_int_cal],
        ["MAPE%", mape_full_raw, mape_int_raw, mape_full_cal, mape_int_cal, mape_full_raw, mape_int_raw, mape_full_cal,
         mape_int_cal, mape_full_raw, mape_int_raw, mape_full_cal, mape_int_cal],
        ["SMAPE%", smape_full_raw, smape_int_raw, smape_full_cal, smape_int_cal, smape_full_raw, smape_int_raw,
         smape_full_cal, smape_int_cal, smape_full_raw, smape_int_raw, smape_full_cal, smape_int_cal],
    ]

    # --- Header Formatting ---
    header_rows = [
        ["Metric", "RAW GNN (PU)", "", "CALIBRATED (PU)", "", "RAW/CAL (120V)", "", "", "", "RAW/CAL (240V)", "", "",
         ""],
        ["", "Full", f"Int ({d_interval}s)", "Full", f"Int ({d_interval}s)", "RAW Full", "RAW Int", "CAL Full",
         "CAL Int", "RAW Full", "RAW Int", "CAL Full", "CAL Int"]
    ]

    # --- Print Header ---
    col_widths = [10] + [12] * 12


    def print_row(row, widths):
        formatted_row = ""
        for item, width in zip(row, widths):
            formatted_row += f"{item:^{width}}"
        print(formatted_row)


    print("-" * 140)
    print_row(header_rows[0], col_widths)
    print_row(header_rows[1], col_widths)
    print("-" * 140)

    # --- Print Data Rows ---
    for row in data:
        row_name = row[0]
        # Use .6f for absolute errors (AE, RMSE) and .4f for relative errors (NRMSE, MAPE, SMAPE)
        row_values = [f"{v:.6f}" if "AE" in row_name or "RMSE" in row_name else f"{v:.4f}" for v in row[1:]]

        print_row([row_name] + row_values, col_widths)
        if row_name == "MaxAE":
            print("-" * 140)  # Separate Absolute from Relative errors

    print("-" * 140)

    # =================================================================
    # --- END OF FINAL MATRIX SECTION ---
    # =================================================================

    # =================================================================
    # --- 7.5 LAG CHECK (uses final 'node_predictions_full') ---
    # =================================================================
    # NOTE: Old section 7.3, renamed 7.5
    act = node_actuals_full
    pred = node_predictions_full
    K = 10


    def corr_at_lag(a_, p_, l):
        if l > 0:
            aa, pp = a_[:-l], p_[l:]
        elif l < 0:
            aa, pp = a_[-l:], p_[:l]
        else:
            aa, pp = a_, p_
        if len(aa) < 10: return np.nan
        return np.corrcoef(aa, pp)[0, 1]


    lags = range(-K, K + 1)
    cors = np.array([corr_at_lag(act, pred, l) for l in lags], dtype=float)
    best_idx = np.nanargmax(cors)
    best_lag = list(lags)[best_idx]
    print(f"\n[Lag check] best_lag={best_lag} samples, corr={cors[best_idx]:.5f}")

    if best_lag != 0:
        if best_lag > 0:
            a_corr, p_corr = act[:-best_lag], pred[best_lag:]
        else:
            a_corr, p_corr = act[-best_lag:], pred[:best_lag]
        # We use r2_score here just for the debug print
        print(f"[Lag check] R² after lag-correction (debug): {r2_score(a_corr, p_corr):.6f}")

# ---- OUTPUT FOLDER (matches your LaTeX path style) ----
PLOT_DIR = os.path.join("samples", "Figure")
os.makedirs(PLOT_DIR, exist_ok=True)

# =================================================================
# --- 7.6 PLOTTING (uses final 'node_predictions_full') ---
#      ACM single-column friendly + save into a folder
# =================================================================

# ---- OUTPUT FOLDER (matches your LaTeX path style) ----
PLOT_DIR = os.path.join("samples", "Figure")
os.makedirs(PLOT_DIR, exist_ok=True)

# ---- ACM single-column figure size ----
FIG_W, FIG_H = 3.33, 2.30   # inches
DPI = 300

# ---- Fonts / sizes: ACM-ish ----
plt.rcParams.update({
    "font.family": ["Linux Libertine"],
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
    print(f"\n❌ ERROR: Invalid plot slice: start={ps}, end={pe} (need start < end).")
    ps, pe = 0, min(total_len, 800)
    print(f"➡️  Falling back to [0:{pe}).")

node_actuals_plot = node_actuals_full[ps:pe]
node_predictions_plot = node_predictions_full[ps:pe]

interval_indices_plot_global = np.arange(ps, pe, d_interval)
flat_actuals_interval_plot = node_actuals_full[interval_indices_plot_global]
flat_preds_interval_plot = node_predictions_full[interval_indices_plot_global]

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
x_plot = np.arange(ps, pe)

# ---- KEEP ORIGINAL COLORS ----
ax.plot(x_plot, node_predictions_plot, label="Predicted",
        color="red", linestyle="--", linewidth=.7)
ax.plot(x_plot, node_actuals_plot, label="Actual",
        color="blue", linestyle="-", linewidth=.7)

if interval_indices_plot_global.size > 0:
    ax.scatter(interval_indices_plot_global, flat_preds_interval_plot,
               color="orange", s=5, label=f"Predict @ {d_interval} min", zorder=2)
    ax.scatter(interval_indices_plot_global, flat_actuals_interval_plot,
               color="green", s=5, label=f"Actual @ {d_interval} min", zorder=2)

# y-lims + clean ticks
data_lo = float(min(node_actuals_plot.min(), node_predictions_plot.min()))
data_hi = float(max(node_actuals_plot.max(), node_predictions_plot.max()))
yrange = max(1e-9, data_hi - data_lo)
pad = max(0.0002, 0.05 * yrange)
ax.set_ylim(data_lo - pad, data_hi + pad)
ax.yaxis.set_major_locator(MaxNLocator(5))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

ax.xaxis.set_major_locator(MaxNLocator(6))

# add small left/right padding
xspan = max(1, pe - ps)
xpad = 0.03 * xspan
xmin = max(0, ps - xpad)
xmax = (pe - 1) + xpad
ax.set_xlim(xmin, xmax)

ax.set_xlabel(r"$\mathbf{(d)}$ Time step (min)")
ax.set_ylabel("Voltage (p.u.)")

ax.tick_params(direction="in", length=3, width=0.8)
for s in ax.spines.values():
    s.set_linewidth(0.8)

leg = ax.legend(loc="best", frameon=True, framealpha=1.0,
                facecolor="white", edgecolor="0.3",
                borderpad=0.25, labelspacing=0.25, handlelength=2.0)
leg.get_frame().set_linewidth(0.6)

ax.grid(False)
fig.subplots_adjust(left=0.14, right=0.995, bottom=0.22, top=0.98)

# ---- SAVE (same naming style as your LaTeX) ----
base = os.path.join(PLOT_DIR, f"{NODE_TO_PLOT_ID}{PHASE}")  # e.g., 180C
fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(base + ".png", dpi=DPI, bbox_inches="tight", pad_inches=0.01)
plt.show()
plt.close(fig)

print(f"Saved: {base}.pdf and {base}.png")


# =================================================================
# --- 8. RESIDUAL PLOTS ---
# =================================================================
print("\n--- Generating Residual Plots ---")

residuals_raw = node_actuals_full - node_predictions_full_raw
residuals_cal = node_actuals_full - node_predictions_full

fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
x_plot = np.arange(ps, pe)

# ---- KEEP ORIGINAL COLORS ----
if calibration_was_run and calibrator_node is not None:
    ax.plot(x_plot, residuals_cal[ps:pe],
            label=f"Calibrated (MAE {mae_full_cal:.4f})",
            color="purple", linewidth=1.2)
    ax.plot(x_plot, residuals_raw[ps:pe],
            label=f"Raw (MAE {mae_full_raw:.4f})",
            color="orange", linestyle="--", linewidth=1.0, alpha=0.85)
else:
    ax.plot(x_plot, residuals_raw[ps:pe],
            label=f"Raw (MAE {mae_full_raw:.4f})",
            color="purple", linewidth=1.2)

ax.axhline(0, color="black", linestyle="--", linewidth=0.9, label="Zero")

ax.xaxis.set_major_locator(MaxNLocator(6))
ax.yaxis.set_major_locator(MaxNLocator(5))

# same padding
xspan = max(1, pe - ps)
xpad = 0.03 * xspan
xmin = max(0, ps - xpad)
xmax = (pe - 1) + xpad
ax.set_xlim(xmin, xmax)

ax.set_xlabel("Time step (min)")
ax.set_ylabel("Residual")

ax.tick_params(direction="in", length=3, width=0.8)
for s in ax.spines.values():
    s.set_linewidth(0.8)

leg = ax.legend(loc="best", frameon=True, framealpha=1.0,
                facecolor="white", edgecolor="0.3",
                borderpad=0.25, labelspacing=0.25, handlelength=2.0)
leg.get_frame().set_linewidth(0.6)

ax.grid(False)
fig.subplots_adjust(left=0.14, right=0.995, bottom=0.22, top=0.98)

base = os.path.join(PLOT_DIR, f"{NODE_TO_PLOT_ID}{PHASE}_residual")  # e.g., 180C_residual
fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(base + ".png", dpi=DPI, bbox_inches="tight", pad_inches=0.01)
plt.show()
plt.close(fig)

print(f"Saved: {base}.pdf and {base}.png")
print("--- Residual plots generated ---")
print("\n--- Evaluation Script Finished ---")
