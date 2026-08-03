# Voltage_loader.py (Corrected)

import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

class VoltageDatasetLoader:
    """
    A robust, universal data loader that can handle Phases A, B, and C.
    It automatically detects the data structure for the specified phase
    and applies the correct preprocessing steps.
    """
    def __init__(self, lags: int = 60, forecast_horizon: int = 12, stride: int = 1, data_dir: str = ".", phase: str = 'A'):
        self.lags = lags
        self.forecast_horizon = forecast_horizon
        self.stride = stride
        self._data_dir = data_dir
        self.phase = phase.upper()
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        """Private method to run all data processing steps."""
        print(f"➡️ Step 1: Loading raw data files for Phase {self.phase}...")
        try:
            voltage_path = os.path.join(self._data_dir, f'MVtrue_Vmag_phase{self.phase}.csv')
            features = pd.read_csv(voltage_path, header=None).iloc[:, 1:].values

            adj_path = os.path.join(self._data_dir, f'Conn_phase{self.phase}.csv')
            adj_matrix_raw = pd.read_csv(adj_path, header=None).values
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ ERROR: Could not find '{e.filename}'. Ensure all phase files are present.")

        num_feature_nodes = features.shape[1]
        if self.phase in ['A', 'B'] and adj_matrix_raw.shape[0] > num_feature_nodes:
             print(f"    ...Phase {self.phase} mismatch detected. Filtering adjacency matrix.")
             missing_node_index = 29
             adj_matrix_temp = np.delete(adj_matrix_raw, missing_node_index, axis=0)
             self.adj_matrix = np.delete(adj_matrix_temp, missing_node_index, axis=1)
        else:
            print(f"    ...Adjacency matrix dimensions match feature data.")
            self.adj_matrix = adj_matrix_raw

        print("➡️ Step 2: Splitting data chronologically (70/10/20)...")
        n_timesteps = features.shape[0]
        train_split = int(n_timesteps * 0.7)
        val_split = int(n_timesteps * 0.8)
        train_data = features[:train_split]
        val_data = features[train_split:val_split]
        test_data = features[val_split:]

        print("➡️ Step 3: Normalizing data with node-wise Z-Score...")
        self.mean = np.mean(train_data, axis=0)
        self.std = np.std(train_data, axis=0)
        self.std[self.std == 0] = 1e-6
        train_norm = (train_data - self.mean) / self.std
        val_norm = (val_data - self.mean) / self.std
        test_norm = (test_data - self.mean) / self.std
        print(f"    ...Computed unique mean and std for all {train_data.shape[1]} nodes.")

        print("➡️ Step 4: Creating lagged sequences for forecasting...")
        self.train_X, self.train_Y = self._create_lags(train_norm)
        self.val_X, self.val_Y = self._create_lags(val_norm)
        self.test_X, self.test_Y = self._create_lags(test_norm)
        print(f"    ...Created {self.train_X.shape[0]} training samples.")

        print("➡️ Step 5: Adding 'feature' dimension...")
        self.train_X = np.expand_dims(self.train_X, axis=-1)
        self.train_Y = np.expand_dims(self.train_Y, axis=-1)
        self.val_X = np.expand_dims(self.val_X, axis=-1)
        self.val_Y = np.expand_dims(self.val_Y, axis=-1)
        self.test_X = np.expand_dims(self.test_X, axis=-1)
        self.test_Y = np.expand_dims(self.test_Y, axis=-1)
        print(f"    ...Final train_X shape: {self.train_X.shape}")
        print(f"    ...Final train_Y shape: {self.train_Y.shape}")


    def _create_lags(self, data):
        """
        Creates samples where X is a sequence of past steps (lags)
        and Y is a sequence of future steps (forecast_horizon).
        """
        X, Y = [], []
        total_length = self.lags + self.forecast_horizon
        for i in range(0, len(data) - total_length + 1, self.stride):
            input_end = i + self.lags
            X.append(data[i:input_end])
            output_end = input_end + self.forecast_horizon
            Y.append(data[input_end:output_end])
        return np.array(X), np.array(Y)

    def get_dataloaders(self, batch_size: int = 32, shuffle: bool = False):
        """Creates and returns PyTorch DataLoader objects."""
        print("➡️ Step 6: Creating PyTorch DataLoaders...")
        train_dataset = TensorDataset(torch.from_numpy(self.train_X).float(), torch.from_numpy(self.train_Y).float())
        val_dataset = TensorDataset(torch.from_numpy(self.val_X).float(), torch.from_numpy(self.val_Y).float())
        test_dataset = TensorDataset(torch.from_numpy(self.test_X).float(), torch.from_numpy(self.test_Y).float())

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        print("✅ All data processed and batched successfully.")
        adj_matrix_t = torch.from_numpy(self.adj_matrix).float()
        mean_t = torch.tensor(self.mean).float()
        std_t = torch.tensor(self.std).float()

        return train_loader, val_loader, test_loader, adj_matrix_t, mean_t, std_t