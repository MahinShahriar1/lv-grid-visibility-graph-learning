import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
import csv
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch.optim as optim
import pandas as pd # ADDED: Required for saving error matrices
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
# --- 1. IMPORT THE CORRECT LOADER ---
try:
    from Loader_File2 import StaticVoltageLoader # type: ignore
except ImportError:
    print("=" * 50)
    print("❌ ERROR: Could not find 'Loader_File.py'")
    print("Please save the StaticVoltageLoader class into a file named 'Loader_File.py' in the same directory.")
    print("=" * 50)
    exit()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"--- Using device: {device} ---")

# =================================================================
# --- EARLY STOPPING CLASS ---
# =================================================================
class EarlyStopping:
    def __init__(self, patience=20, min_delta=0.0001, path='best_model_static_graphsage.pt'):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.path = path
        print(f"[EarlyStopping] Initialized with patience={patience}, min_delta={min_delta}")

    def check(self, val_loss, model):
        if self.best_loss is None:
            print(f"  [EarlyStopping] Initializing best_loss to {val_loss:.6f}.")
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
        elif val_loss < self.best_loss - self.min_delta:
            print(f"  [EarlyStopping] Validation loss improved ({self.best_loss:.6f} --> {val_loss:.6f}). Saving model and resetting counter.")
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        else:
            self.counter += 1
            print(f"  [EarlyStopping] Validation loss did not improve from {self.best_loss:.6f}. Counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        torch.save(model.state_dict(), self.path)

# =================================================================
# --- 2. WEIGHTED SAGE CONV CLASS ---
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

        # 4) GNN layers (same logic, just using PyG layer)
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
# --- 4. DATA LOADING ---
# =================================================================
print("--- Loading Custom Voltage Data ---")
PHASE = 'A'
HRRSM_NODES = [1, 3, 8, 16, 20, 25, 28, 31, 44, 52, 57, 58, 66, 79, 85, 97, 106, 123, 136, 149, 169, 201, 204, 206, 212,
               231, 234, 237, 249, 253, 265, 269, 277, 280, 282, 283, 289, 290, 293, 299, 313, 315, 322, 327, 332, 333,
               339, 344, 346, 349, 351, 365, 371, 374, 376, 379, 383, 398, 405, 409, 425, 437, 443, 450, 451, 452, 455]
BATCH_SIZE = 32


loader = StaticVoltageLoader(data_dir=".", phase=PHASE, hrrsm_node_list=HRRSM_NODES)
train_loader, val_loader, test_loader, adj_matrix_t, mean_t, std_t = loader.get_dataloaders(batch_size=BATCH_SIZE, shuffle=True)

adj_matrix_t = adj_matrix_t.to(device)
mean_t = mean_t.to(device)
std_t = std_t.to(device)

num_all_nodes = len(loader.all_nodes_list_str)
hrrsm_indices = torch.tensor(loader.hrrsm_indices).long().to(device)
sm_indices = torch.tensor(loader.sm_indices).long().to(device)

# Unscale helpers (per SM node)
sm_means = mean_t[sm_indices].cpu()
sm_stds  = std_t[sm_indices].cpu()

# Unscaling utility function (moved outside of the deleted calibration block)
def _unscale(tensor_on_cpu):
    return (tensor_on_cpu * sm_stds) + sm_means  # broadcast per node

print("--- Creating PyG edge_index and edge_weight ---")
src_nodes, dst_nodes = adj_matrix_t.nonzero(as_tuple=True)  # both [E]
edge_index = torch.stack([src_nodes, dst_nodes], dim=0).long().to(device)  # [2, E]
edge_weight = adj_matrix_t[src_nodes, dst_nodes].float().to(device)        # [E]
print(f"  Graph tensors created with {num_all_nodes} nodes and {edge_index.size(1)} edges.")

model = StaticGraphSAGE(
    in_feats=1,
    hidden_feats=256,
    out_feats=1,
    num_all_nodes=num_all_nodes,
    hrrsm_indices=hrrsm_indices,
    sm_indices=sm_indices,
    num_gnn_layers=3,
    dropout=0.2
).to(device)



optimizer = optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6)

save_path = f'best_model_static_graphsage6pyG_Phase{PHASE}.pt'
early_stopper = EarlyStopping(patience=20, path=save_path)
num_epochs = 1000


LOG_DIR = "training_logs2"
os.makedirs(LOG_DIR, exist_ok=True)

METRICS_CSV = os.path.join(LOG_DIR, f"metrics_train_val_Phase{PHASE}.csv")

PLOT_ONLY = False
if PLOT_ONLY:
    plot_train_val_from_csv(METRICS_CSV, PHASE, out_dir=LOG_DIR, logy=True) # type: ignore
    raise SystemExit

# Create/overwrite CSV with header at start of training
with open(METRICS_CSV, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["epoch", "train_mse", "val_mse", "lr"])
print(f"Logging metrics to: {METRICS_CSV}")
# =================================================================
# --- 6. TRAIN ---
# =================================================================
train_mse_per_epoch, val_mse_per_epoch = [], []
print("--- Starting Model Training ---")
for epoch in range(num_epochs):
    model.train()
    total_train_mse = 0.0
    for x_input, y_target in train_loader:
        x_input = x_input.to(device); y_target = y_target.to(device)
        optimizer.zero_grad()
        y_hat = model(x_input, edge_index, edge_weight)
        loss = criterion(y_hat, y_target)
        loss.backward(); optimizer.step()
        total_train_mse += loss.item()

    avg_train_mse = total_train_mse / len(train_loader)
    train_mse_per_epoch.append(avg_train_mse)

    # validation
    model.eval()
    total_val_mse = 0.0
    with torch.no_grad():
        for x_input, y_target in val_loader:
            x_input = x_input.to(device); y_target = y_target.to(device)
            y_hat = model(x_input, edge_index, edge_weight)
            total_val_mse += criterion(y_hat, y_target).item()

    avg_val_mse = total_val_mse / len(val_loader)
    val_mse_per_epoch.append(avg_val_mse)

    scheduler.step(avg_val_mse)

    current_lr = optimizer.param_groups[0]["lr"]

    with open(METRICS_CSV, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([epoch + 1, avg_train_mse, avg_val_mse, current_lr])

    early_stopper.check(avg_val_mse, model)
    print(f"Epoch {epoch+1}/{num_epochs} | Train MSE: {avg_train_mse:.6f} | Val MSE: {avg_val_mse:.6f} | LR: {scheduler.optimizer.param_groups[0]['lr']:.6f}")
    if early_stopper.early_stop:
        print(f"🛑 Early stopping triggered at Epoch {epoch + 1}."); break

# [NOTE: The old 6.5 VALIDATION-BASED CALIBRATION block has been removed]

# =================================================================
# --- 7. TEST / EVALUATION ---
# =================================================================
print("\n------ Test Set Evaluation (Uncalibrated) ------")
if early_stopper.best_loss is not None:
    print(f"Loading best model from {early_stopper.path}")
    model.load_state_dict(torch.load(early_stopper.path, weights_only=True))

model.eval()
all_predictions, all_actuals = [], []
with torch.no_grad():
    for x_input, y_target in test_loader:
        x_input = x_input.to(device)
        y_hat = model(x_input, edge_index, edge_weight)
        all_predictions.append(y_hat.cpu())
        all_actuals.append(y_target.cpu())

predictions_tensor = torch.cat(all_predictions)   # [T, Ns]
actuals_tensor     = torch.cat(all_actuals)       # [T, Ns]

# --- Unscale ---
predictions_unscaled = _unscale(predictions_tensor).numpy()
actuals_unscaled     = _unscale(actuals_tensor).numpy()

# --- Report Global Metrics (Full Test Set) ---
flat_predictions = predictions_unscaled.flatten()
flat_actuals     = actuals_unscaled.flatten()

# Global MAE, RMSE, R2 for all time steps and all nodes
mae_full = mean_absolute_error(flat_actuals, flat_predictions)
rmse_full = np.sqrt(mean_squared_error(flat_actuals, flat_predictions))
r2_full = r2_score(flat_actuals, flat_predictions)

print("\n--- 1. Overall Metrics (Full Test Set) ---")
print(f"  MAE:  {mae_full:.6f}")
print(f"  RMSE: {rmse_full:.6f}")
print(f"  R²:   {r2_full:.6f}")
print("-" * 50)

# =======================================================
# --- DETAILED ERROR STATISTICS (Derived from Full Matrices) ---
# =======================================================

# Absolute Error Matrix (T x Ns)
ae_matrix_full = np.abs(actuals_unscaled - predictions_unscaled)

# Time-Wise Average AE (Vector of length T) - Average loss across all nodes per time step
time_wise_avg_ae = np.mean(ae_matrix_full, axis=1)

print("--- 2. Error Statistics (Derived from Full AE Matrix) ---")
print(f"  Mean Time-Wise AE (Avg loss across all nodes): {time_wise_avg_ae.mean():.6f}")
print(f"  Max Time-Wise AE (Worst time step, avg across nodes): {time_wise_avg_ae.max():.6f}")
print("-" * 50)


# =======================================================
# --- METRICS ON 15-MINUTE SUBSET (Every 15th Timestep) ---
# =======================================================
d_interval = 15
n_steps = actuals_unscaled.shape[0]
interval_idx = np.arange(0, n_steps, d_interval)

actuals_15        = actuals_unscaled[interval_idx, :]
predictions_15    = predictions_unscaled[interval_idx, :]

flat_actuals_15     = actuals_15.flatten()
flat_predictions_15 = predictions_15.flatten()

# Global MAE, RMSE, R2 for the 15-minute subset
mae_15 = mean_absolute_error(flat_actuals_15, flat_predictions_15)
rmse_15 = np.sqrt(mean_squared_error(flat_actuals_15, flat_predictions_15))
r2_15 = r2_score(flat_actuals_15, flat_predictions_15)

print("--- 3. Metrics on 15-min Subset (Every 15th Timestep) ---")
print(f"  MAE:  {mae_15:.6f}")
print(f"  RMSE: {rmse_15:.6f}")
print(f"  R²:   {r2_15:.6f}")

# Detailed AE Statistics for 15-minute Subset
ae_matrix_15 = np.abs(actuals_15 - predictions_15)
node_wise_avg_ae_15 = np.mean(ae_matrix_15, axis=0)

print("\n--- 4. Error Statistics (Derived from 15-min Subset AE Matrix) ---")
print(f"  Mean Node-Wise AE (Avg loss across 15-min samples): {node_wise_avg_ae_15.mean():.6f}")
print(f"  Max Node-Wise AE (Worst node, avg across 15-min samples): {node_wise_avg_ae_15.max():.6f}")
print("----------------------------------------------------------")
# =================================================================
# --- 8. VISUALIZATION ---
# =================================================================
NODE_TO_PLOT_ID = 22
TIME_STEPS_TO_PLOT = 200
# reuse d_interval = 15 for plotting highlights

SM_NODE_IDS = [int(loader.all_nodes_list_str[i]) for i in loader.sm_indices]
try:
    node_plot_index = SM_NODE_IDS.index(NODE_TO_PLOT_ID)
except ValueError:
    print(f"\n❌ ERROR: Node {NODE_TO_PLOT_ID} is not a valid SM node.")
    print(f"Choose from: {SM_NODE_IDS}")
    node_plot_index = -1

if node_plot_index != -1:
    # training curves
    plt.figure(figsize=(12, 5))
    plt.plot(train_mse_per_epoch, label='Train MSE', linewidth=2)
    plt.plot(val_mse_per_epoch, label='Validation MSE', linewidth=2)
    plt.title(f'Phase {PHASE} - Training and Validation MSE')
    plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.legend(); plt.grid(True)
    plt.tight_layout(); plt.savefig(f'static_model_loss_plot_Phase{PHASE}.png'); plt.show()

    # scatter (uncalibrated)
    plt.figure(figsize=(8, 8))
    plt.scatter(flat_actuals, flat_predictions, alpha=0.2, s=10, color='tab:blue')
    vmin = min(flat_actuals.min(), flat_predictions.min())
    vmax = max(flat_actuals.max(), flat_predictions.max())
    plt.plot([vmin, vmax], [vmin, vmax],
             linestyle='--', linewidth=2,
             label='Perfect', color='red')
    plt.title(f'Phase {PHASE} - Predicted vs Actual (Uncalibrated, All SM Nodes)')
    plt.xlabel('Actual (Unscaled)'); plt.ylabel('Predicted (Unscaled)')
    plt.legend(); plt.axis('equal'); plt.grid(True); plt.tight_layout()
    plt.savefig(f'static_model_scatter_plot_Phase{PHASE}.png'); plt.show()

    # node series (uncalibrated)
    node_actuals_full = actuals_unscaled[:, node_plot_index]
    node_predictions_full = predictions_unscaled[:, node_plot_index]
    node_actuals = node_actuals_full[:TIME_STEPS_TO_PLOT]
    node_predictions = node_predictions_full[:TIME_STEPS_TO_PLOT]
    interval_indices = np.arange(0, len(node_actuals), d_interval)
    flat_actuals_interval = node_actuals[interval_indices]
    flat_preds_interval   = node_predictions[interval_indices]

    plt.figure(figsize=(15, 6))
    plt.plot(node_predictions, label='Forecasted (Uncalibrated)', linestyle='--', linewidth=2)
    plt.plot(node_actuals,    label='Actual (Ground Truth)',  linewidth=2)
    plt.scatter(interval_indices, flat_preds_interval,  s=30, label=f'Forecast @ {d_interval}', zorder=5)
    plt.scatter(interval_indices, flat_actuals_interval, s=30, label=f'Actual @ {d_interval}',  zorder=5)
    dmin = float(min(node_actuals.min(), node_predictions.min()))
    dmax = float(max(node_actuals.max(), node_predictions.max()))
    pad = max(1e-3, 0.01 * (dmax - dmin + 1e-6))
    plt.ylim(dmin - pad, dmax + pad)

    plt.title(f'Phase {PHASE} - Node {NODE_TO_PLOT_ID} Forecast vs Actual (First {TIME_STEPS_TO_PLOT} Samples)')
    plt.xlabel('Sample Index'); plt.ylabel('Voltage Magnitude (Unscaled)')
    plt.legend(); plt.grid(True); plt.tight_layout()
    plt.savefig(f'static_model_node{NODE_TO_PLOT_ID}_forecast_vs_actual_Phase{PHASE}.png', dpi=300)
    plt.show()


# ---- 2-column ACM: single-column figure size ----
FIG_W, FIG_H = 3.45, 2.35   # inches (good for \columnwidth)
DPI = 600

# ---- Fonts: Libertine ----
plt.rcParams.update({
    "font.family": ["Linux Libertine"],
    "font.size": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

def plot_train_val_from_csv(METRICS_CSV: str, PHASE: str, out_dir: str = "training_logs", logy: bool = True):
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(METRICS_CSV)
    if "epoch" not in df.columns:
        df["epoch"] = range(1, len(df) + 1)

    epochs = df["epoch"].to_numpy()
    train_mse = df["train_mse"].to_numpy()
    val_mse   = df["val_mse"].to_numpy()

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=300)
    ax.plot(epochs, train_mse, label="Train", linewidth=1.6)
    ax.plot(epochs, val_mse,   label="Val", linewidth=1.6, linestyle="--")

    if logy:
        ax.set_yscale("log")

    ax.xaxis.set_major_locator(MaxNLocator(6))
    ax.yaxis.set_major_locator(MaxNLocator(5))

    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE")

    ax.tick_params(direction="in", length=3, width=0.8)
    for s in ax.spines.values():
        s.set_linewidth(0.9)

    leg = ax.legend(loc="best", frameon=True, framealpha=1.0,
                    facecolor="white", edgecolor="0.3",
                    borderpad=0.3, labelspacing=0.3, handlelength=2.0)
    leg.get_frame().set_linewidth(0.6)

    ax.grid(False)
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.18, top=0.98)

    base = os.path.join(out_dir, f"train_val_mse_vs_epoch_Phase{PHASE}")
    fig.savefig(base + ".pdf", bbox_inches="tight")
    fig.savefig(base + ".png", dpi=DPI, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    print(f"Saved: {base}.pdf and {base}.png")
