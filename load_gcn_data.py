import os
import numpy as np
import pandas as pd
import torch
from typing import Optional
from torch.utils.data import TensorDataset, DataLoader


class StaticVoltageLoader:
    """
    Static regression loader (Python 3.9 / headerless voltage CSV).

    - Reads nodes list from nodesMVLV_phase*.csv (header=None, first row = IDs)
    - Reads voltage CSV with header=None and assigns node IDs as column names
    - Contiguous split 70/10/20; fit μ,σ on TRAIN only (leak-free)
    - HRRSM/SM split by provided node list (IDs normalized)
    - Optional correlation-matrix hygiene (symmetrize, zero diag, abs-threshold)
    """

    def __init__(
        self,
        data_dir: str = ".",
        phase: str = "A",
        hrrsm_node_list: list = (1, 4, 11, 15, 21, 25, 31),
        *,
        force_symmetric: bool = True,
        zero_diag: bool = True,
        corr_abs_threshold: Optional[float] = None  # e.g., 0.05 to sparsify
    ):
        self._data_dir = data_dir
        self.phase = phase.upper()
        self.force_symmetric = force_symmetric
        self.zero_diag = zero_diag
        self.corr_abs_threshold = corr_abs_threshold

        print(f"--- Initializing Static Loader for Phase {self.phase} (PURE Correlation) ---")

        self.nodes_path   = os.path.join(self._data_dir, f"nodesMVLV_phase{self.phase}.csv")
        self.voltage_path = os.path.join(self._data_dir, f"MVLV_VmagTure_phase{self.phase}.csv")
        self.adj_path     = os.path.join(self._data_dir, f"ConnMVLV_phase{self.phase}.csv")

        # Normalize IDs: cast floats that are *integers* (e.g., '357.0'→'357'), keep non-integer floats as-is.
        def norm_id(x):
            s = str(x).strip()
            try:
                v = float(s)
                if abs(v - round(v)) < 1e-9:
                    return str(int(round(v)))
                return s
            except Exception:
                return s
        self._norm_id = norm_id

        # 1) Load node list (headerless CSV: first row contains IDs)
        print(f"➡️ Loading node list for Phase {self.phase}...")
        try:
            nodes_df = pd.read_csv(self.nodes_path, header=None)
        except Exception as e:
            raise RuntimeError(f"❌ ERROR: Failed to read node file '{self.nodes_path}': {e}")

        raw_ids = nodes_df.iloc[0].tolist()
        self.all_nodes_list_str = [norm_id(x) for x in raw_ids]
        print(f"  ...Found {len(self.all_nodes_list_str)} total nodes for Phase {self.phase}.")

        # HRRSM vs SM split
        conceptual_hrrsm_nodes_str = {norm_id(n) for n in hrrsm_node_list}
        self.hrrsm_nodes_str = [n for n in self.all_nodes_list_str if n in conceptual_hrrsm_nodes_str]
        self.sm_nodes_str    = [n for n in self.all_nodes_list_str if n not in conceptual_hrrsm_nodes_str]

        if not self.hrrsm_nodes_str:
            print("  ...WARNING: No HRRSM nodes from your list were found in nodes CSV (check ID formatting).")

        self.hrrsm_indices = [self.all_nodes_list_str.index(n) for n in self.hrrsm_nodes_str]
        self.sm_indices    = [self.all_nodes_list_str.index(n) for n in self.sm_nodes_str]

        # 2) Load data & preprocess
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        print("➡️ Step 1: Loading COMPLETE data files...")

        # Voltage CSV: headerless, assign node IDs (and optional Time_min)
        try:
            df = pd.read_csv(self.voltage_path, header=None)
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ ERROR: Could not find voltage file. {e}")

        # If first column is time, expect nodes+1 columns; else expect exactly nodes columns.
        if df.shape[1] == len(self.all_nodes_list_str) + 1:
            df.columns = ["Time_min"] + self.all_nodes_list_str
            df = df.drop(columns="Time_min")
        elif df.shape[1] == len(self.all_nodes_list_str):
            df.columns = self.all_nodes_list_str
        else:
            raise ValueError(
                f"❌ Voltage CSV has {df.shape[1]} columns, but nodes list has {len(self.all_nodes_list_str)} "
                f"(or {len(self.all_nodes_list_str)+1} with a time column)."
            )

        # Ensure order exactly matches nodes list (safety)
        df = df[self.all_nodes_list_str]
        features = df.values.astype(np.float32)  # [N_samples, N_nodes]
        n_nodes = features.shape[1]

        # Adjacency / correlation matrix
        try:
            A = pd.read_csv(self.adj_path, header=None).values.astype(np.float32)
            print("  ...Loaded PURE Correlation Matrix.")
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ ERROR: Could not find correlation file. {e}")

        if A.shape != (n_nodes, n_nodes):
            print(f"  ...WARNING: Adj matrix shape {A.shape} -> cropping to ({n_nodes}, {n_nodes}).")
            A = A[:n_nodes, :n_nodes]

        if self.force_symmetric:
            A = 0.5 * (A + A.T)
        if self.zero_diag:
            np.fill_diagonal(A, 0.0)
        if self.corr_abs_threshold is not None:
            A = np.where(np.abs(A) >= float(self.corr_abs_threshold), A, 0.0)

        self.adj_matrix = A

        # Step 2: contiguous split, fit μ/σ on TRAIN only
        print("➡️ Step 2: Contiguous split (70/10/20) and leak-free normalization (fit on TRAIN only)...")
        N = features.shape[0]
        n_train = int(0.7 * N)
        n_val   = int(0.1 * N)
        idx_train = np.arange(0, n_train)
        idx_val   = np.arange(n_train, n_train + n_val)
        idx_test  = np.arange(n_train + n_val, N)

        mu = features[idx_train].mean(axis=0).astype(np.float32)  # [n_nodes]
        sigma = features[idx_train].std(axis=0).astype(np.float32)
        sigma[sigma == 0] = 1e-6

        features_norm = (features - mu) / sigma

        # X (HRRSM) / Y (SM)
        X_norm = features_norm[:, self.hrrsm_indices]
        Y_norm = features_norm[:, self.sm_indices]

        self.train_X, self.train_Y = X_norm[idx_train], Y_norm[idx_train]
        self.val_X,   self.val_Y   = X_norm[idx_val],   Y_norm[idx_val]
        self.test_X,  self.test_Y  = X_norm[idx_test],  Y_norm[idx_test]

        print(f"  ...Train samples: {self.train_X.shape[0]}")
        print(f"  ...Val samples:   {self.val_X.shape[0]}")
        print(f"  ...Test samples:  {self.test_X.shape[0]}")

        # Save μ,σ for per-node de-normalization later (aligned to all_nodes_list_str)
        self.mean = mu
        self.std  = sigma

    def get_dataloaders(self, batch_size: int = 32, shuffle: bool = True):
        print("➡️ Step 3: Creating PyTorch DataLoaders...")

        train_ds = TensorDataset(torch.from_numpy(self.train_X).float(),
                                 torch.from_numpy(self.train_Y).float())
        val_ds   = TensorDataset(torch.from_numpy(self.val_X).float(),
                                 torch.from_numpy(self.val_Y).float())
        test_ds  = TensorDataset(torch.from_numpy(self.test_X).float(),
                                 torch.from_numpy(self.test_Y).float())

        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

        print("✅ All data processed and batched successfully.")

        adj_matrix_t = torch.from_numpy(self.adj_matrix).float()
        mean_t = torch.tensor(self.mean).float()
        std_t  = torch.tensor(self.std).float()
        return train_loader, val_loader, test_loader, adj_matrix_t, mean_t, std_t
