import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import scatter
import csv
import random
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import torch.optim as optim
import pandas as pd # ADDED: Required for saving error matrices
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
import argparse
from torch_geometric.nn import GCNConv
parser = argparse.ArgumentParser()
parser.add_argument("--phase", choices=["A","B","C"], required=True)
args = parser.parse_args()
PHASE = args.phase

# --- 1. IMPORT THE CORRECT LOADER ---
try:
    from Load_GCN import StaticVoltageLoader  # type: ignore
except ImportError:
    print("=" * 50)
    print("❌ ERROR: Could not find 'Load_GCN.py' (StaticVoltageLoader).")
    print("Make sure Load_GCN.py is in the same folder, or on PYTHONPATH.")
    print("=" * 50)
    raise


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
class StaticGCN(nn.Module):
    def __init__(self, in_feats, hidden_feats, out_feats,
                 num_all_nodes, hrrsm_indices, sm_indices,
                 num_gnn_layers=2, dropout=0.2):
        super().__init__()
        self.num_all_nodes = num_all_nodes
        self.hrrsm_indices = hrrsm_indices
        self.sm_indices = sm_indices
        self.hidden_feats = hidden_feats

        # Vanilla GCN layers (NO edge weights used)
        self.gnn_layers = nn.ModuleList([GCNConv(in_feats, hidden_feats)])
        for _ in range(num_gnn_layers - 1):
            self.gnn_layers.append(GCNConv(hidden_feats, hidden_feats))

        self.dropout = nn.Dropout(dropout)

        # Keep your MLP head exactly in spirit
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_feats, hidden_feats // 2),
            nn.ReLU(),
            nn.Linear(hidden_feats // 2, out_feats)
        )

    def forward(self, x_input_batch, edge_index, edge_weight=None):
        """
        Kept signature (edge_weight arg) so you don't have to change the rest of your script.
        edge_weight is ignored to keep it "vanilla" / unweighted GCN.

        x_input_batch: [B, num_HRRSM_nodes]
        edge_index: [2, E] for the single feeder graph (no batch)
        """
        device = x_input_batch.device
        batch_size = x_input_batch.shape[0]

        # 1) Put HRRSM measurements into full node feature tensor
        x_full = torch.zeros(batch_size, self.num_all_nodes, 1, device=device)
        x_full[:, self.hrrsm_indices, 0] = x_input_batch

        # 2) Flatten to [B*N, F]
        h = x_full.view(batch_size * self.num_all_nodes, 1)

        # 3) Build batched edge_index by offsetting node IDs
        N = self.num_all_nodes
        E = edge_index.size(1)

        if batch_size == 1:
            edge_index_batch = edge_index
        else:
            edge_index_batch = edge_index.repeat(1, batch_size)           # [2, B*E]
            offsets = torch.arange(batch_size, device=device).repeat_interleave(E) * N
            edge_index_batch = edge_index_batch + offsets                 # broadcast to both rows

        # 4) GCN layers (vanilla)
        for i, layer in enumerate(self.gnn_layers):
            h = layer(h, edge_index_batch)    # NOTE: no edge_weight passed
            h = F.relu(h)
            if i < len(self.gnn_layers) - 1:
                h = self.dropout(h)

        # 5) Reshape back to [B, N, hidden_feats]
        h_full_nodes = h.view(batch_size, self.num_all_nodes, self.hidden_feats)

        # 6) Select SM nodes and predict with your MLP head
        h_target_nodes = h_full_nodes[:, self.sm_indices, :]   # [B, Ns, hidden_feats]
        out = self.prediction_head(h_target_nodes)             # [B, Ns, 1]
        return out.squeeze(-1)                                # [B, Ns]

# =================================================================
# --- 4. DATA LOADING ---
# =================================================================
print("--- Loading Custom Voltage Data ---")
print(f"[ARGPARSE] phase = {PHASE}")
# PHASE = 'A'
# HRRSM_NODES = [1, 3, 8, 16, 20, 25, 28, 31, 44, 52, 57, 58, 66, 79, 85, 97, 106, 123, 136, 149, 169, 201, 204, 206, 212,
#                231, 234, 237, 249, 253, 265, 269, 277, 280, 282, 283, 289, 290, 293, 299, 313, 315, 322, 327, 332, 333,
#                339, 344, 346, 349, 351, 365, 371, 374, 376, 379, 383, 398, 405, 409, 425, 437, 443, 450, 451, 452, 455]
random.seed(42)
HRRSM_NODES = sorted(random.sample(range(1, 460), 67))
print("HRRSM_NODES (count = {}):".format(len(HRRSM_NODES)))
print(HRRSM_NODES)
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

model = StaticGCN(
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

save_path = f'best_model_static_gcn_random_pyG_Phase{PHASE}.pt'
early_stopper = EarlyStopping(patience=20, path=save_path)
num_epochs = 1000


LOG_DIR = "training_logs_test_gcn+random"
os.makedirs(LOG_DIR, exist_ok=True)

METRICS_CSV = os.path.join(LOG_DIR, f"metrics_train_val_GCN_Random_Phase{PHASE}.csv")

PLOT_ONLY = False

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

