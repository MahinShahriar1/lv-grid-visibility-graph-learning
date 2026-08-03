# custom_loader.py

import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader


class VoltageDatasetLoader:
    """
    A custom data loader designed to mimic the style of standard graph
    time series loaders like PEMS-BAY.

    It is initialized with the number of `lags` (the lookback window) and
    prepares the data for a recurrent model. Its main method, `get_dataloaders`,
    returns ready-to-use PyTorch DataLoader objects.
    """

    def __init__(self, lags: int = 60, data_dir: str = "."):
        self.lags = lags
        self._data_dir = data_dir
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        """Private method to run all data processing steps."""
        print("➡️ Step 1: Loading raw data files...")
        # --- Load raw data ---
        try:
            voltage_path = os.path.join(self._data_dir, 'MVtrue_Vmag_phaseA.csv')
            features = pd.read_csv(voltage_path, header=None).iloc[:, 1:].values
            adj_path = os.path.join(self._data_dir, 'Conn_phaseA.csv')
            adj_matrix_full = pd.read_csv(adj_path, header=None).values
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ ERROR: Could not find '{e.filename}'.")

        # --- Filter adjacency matrix ---
        missing_node_index = 29
        adj_matrix = np.delete(adj_matrix_full, missing_node_index, axis=0)
        self.adj_matrix = np.delete(adj_matrix, missing_node_index, axis=1)

        print("➡️ Step 2: Splitting data chronologically (70/10/20)...")
        # --- Split data ---
        n_timesteps = features.shape[0]
        train_split = int(n_timesteps * 0.7)
        val_split = int(n_timesteps * 0.8)
        train_data = features[:train_split]
        val_data = features[train_split:val_split]
        test_data = features[val_split:]

        print("➡️ Step 3: Normalizing data with Z-Score...")
        # --- Normalize data ---
        self.mean = np.mean(train_data)
        self.std = np.std(train_data)
        train_norm = (train_data - self.mean) / self.std
        val_norm = (val_data - self.mean) / self.std
        test_norm = (test_data - self.mean) / self.std
        print(f"   ...Mean: {self.mean:.4f}, Std Dev: {self.std:.4f}")

        print(f"➡️ Step 4: Creating samples with {self.lags} lags...")
        # --- Create lagged samples ---
        self.train_X, self.train_Y = self._create_lags(train_norm)
        self.val_X, self.val_Y = self._create_lags(val_norm)
        self.test_X, self.test_Y = self._create_lags(test_norm)

        print("➡️ Step 5: Adding 'feature' dimension...")
        # --- Add feature dimension ---
        self.train_X = np.expand_dims(self.train_X, axis=-1)
        self.train_Y = np.expand_dims(self.train_Y, axis=-1)
        self.val_X = np.expand_dims(self.val_X, axis=-1)
        self.val_Y = np.expand_dims(self.val_Y, axis=-1)
        self.test_X = np.expand_dims(self.test_X, axis=-1)
        self.test_Y = np.expand_dims(self.test_Y, axis=-1)
        print(f"   ...Final train_X shape: {self.train_X.shape}")

    def _create_lags(self, data):
        """Creates samples where Y is X shifted by one timestep."""
        X, Y = [], []
        for i in range(len(data) - self.lags):
            X.append(data[i: i + self.lags])
            Y.append(data[i + 1: i + 1 + self.lags])
        return np.array(X), np.array(Y)

    def get_dataloaders(self, batch_size: int = 32, shuffle: bool = True):
        """
        The main public method. Creates and returns PyTorch DataLoader objects
        for training, validation, and testing, along with graph and normalization info.

        Args:
            batch_size (int): The number of samples per batch.
            shuffle (bool): Whether to shuffle the training data.

        Returns:
            A tuple containing:
            (train_loader, val_loader, test_loader, adj_matrix, mean, std)
        """
        print("➡️ Step 6: Creating PyTorch DataLoaders...")

        # Convert numpy arrays to PyTorch tensors
        train_X_t = torch.from_numpy(self.train_X).float()
        train_Y_t = torch.from_numpy(self.train_Y).float()
        val_X_t = torch.from_numpy(self.val_X).float()
        val_Y_t = torch.from_numpy(self.val_Y).float()
        test_X_t = torch.from_numpy(self.test_X).float()
        test_Y_t = torch.from_numpy(self.test_Y).float()

        # Create TensorDatasets
        train_dataset = TensorDataset(train_X_t, train_Y_t)
        val_dataset = TensorDataset(val_X_t, val_Y_t)
        test_dataset = TensorDataset(test_X_t, test_Y_t)

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

        print("✅ All data processed and batched successfully.")

        # Convert graph and stats to tensors
        adj_matrix_t = torch.from_numpy(self.adj_matrix).float()
        mean_t = torch.tensor(self.mean).float()
        std_t = torch.tensor(self.std).float()

        return train_loader, val_loader, test_loader, adj_matrix_t, mean_t, std_t