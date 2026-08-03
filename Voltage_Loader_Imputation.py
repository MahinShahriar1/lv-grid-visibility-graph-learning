# Voltage_loader_imputation.py
# This is a MODIFIED version of your original loader,
# adapted for the new IMUTATION/INFERENCE task.

import pandas as pd
import numpy as np
import os
import torch
from torch.utils.data import TensorDataset, DataLoader

class ImputationVoltageLoader:
    """
    This loader is designed for an IMUTATION task.
    
    - It loads the 'degraded' data (with missing SM values).
    - It creates TRAINING samples only at the 'd_interval' timestamps.
    - It creates TESTING samples for *every* timestamp to fill in the blanks.
    """
    def __init__(self, 
                 lags: int = 15,          # This should be your interval 'd'
                 d_interval: int = 15,    # The reporting rate of Simple Meters
                 data_dir: str = ".", 
                 phase: str = 'A'):
        
        print(f"--- Initializing Imputation Loader ---")
        print(f"  Lags (d): {lags}")
        print(f"  SM Interval: {d_interval}")
        
        self.lags = lags
        self.d_interval = d_interval
        self._data_dir = data_dir
        self.phase = phase.upper()

        # --- Define Node Lists for Phase A (as discussed) ---
        # These are hard-coded for the imputation task
        self.all_nodes_list_str = [str(n) for n in [1, 2, 3, 4, 6, 7, 8, 9, 10, 11, 12, 13, 15, 16, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 34]]
        self.hrrsm_nodes_str = [str(n) for n in [4, 10, 15, 20, 26, 32]]
        self.sm_nodes_str = [n for n in self.all_nodes_list_str if n not in self.hrrsm_nodes_str]
        
        print(f"  Found {len(self.hrrsm_nodes_str)} HRRSM nodes and {len(self.sm_nodes_str)} SM nodes.")
        
        # Get integer indices for numpy slicing
        self.hrrsm_indices = [self.all_nodes_list_str.index(n) for n in self.hrrsm_nodes_str]
        self.sm_indices = [self.all_nodes_list_str.index(n) for n in self.sm_nodes_str]
        
        # Run the processing
        self._load_and_preprocess()

    def _load_and_preprocess(self):
        """Private method to run all data processing steps."""
        
        print(f"➡️ Step 1: Loading DEGRADED data files for Phase {self.phase}...")
        try:
            # --- MODIFIED: Load the degraded data file ---
            voltage_path = os.path.join(self._data_dir, f'degraded_Vmag_phase{self.phase}.csv')
            
            # Load the CSV
            df = pd.read_csv(voltage_path)
            # Sort by time, drop the time column for processing
            df = df.sort_values(by='Time_min').drop(columns='Time_min')
            # Ensure columns are in the correct, consistent order
            df = df[self.all_nodes_list_str]
            
            features = df.values # This is a numpy array with NaNs

            # --- Adjacency matrix loading (unchanged) ---
            adj_path = os.path.join(self._data_dir, f'Conn_phase{self.phase}.csv')
            adj_matrix_raw = pd.read_csv(adj_path, header=None).values
            print("  ...Loaded Adjacency Matrix.")
            
        except FileNotFoundError as e:
            raise FileNotFoundError(f"❌ ERROR: Could not find '{e.filename}'. Ensure all phase files are present.")
        except KeyError as e:
            raise KeyError(f"❌ ERROR: A node in the hard-coded lists is not in the CSV file. {e}")

        # --- Adjacency matrix filtering (unchanged) ---
        num_feature_nodes = features.shape[1]
        if self.phase in ['A', 'B'] and adj_matrix_raw.shape[0] > num_feature_nodes:
            print(f"  ...Phase {self.phase} mismatch detected. Filtering adjacency matrix.")
            missing_node_index = 29
            adj_matrix_temp = np.delete(adj_matrix_raw, missing_node_index, axis=0)
            self.adj_matrix = np.delete(adj_matrix_temp, missing_node_index, axis=1)
        else:
            print(f"  ...Adjacency matrix dimensions match feature data.")
            self.adj_matrix = adj_matrix_raw
            
        print(f"  ...Final matrix shape: {self.adj_matrix.shape}")

        print("➡️ Step 2: Splitting data chronologically (70/10/20)...")
        n_timesteps = features.shape[0]
        train_split = int(n_timesteps * 0.7)
        val_split = int(n_timesteps * 0.8)
        
        train_data = features[:train_split]
        val_data = features[train_split:val_split]
        test_data = features[val_split:]

        print("➡️ Step 3: Normalizing data with node-wise Z-Score (ignoring NaNs)...")
        # --- MODIFIED: Use nanmean and nanstd ---
        self.mean = np.nanmean(train_data, axis=0)
        self.std = np.nanstd(train_data, axis=0)
        self.std[self.std == 0] = 1e-6
        
        train_norm = (train_data - self.mean) / self.std
        val_norm = (val_data - self.mean) / self.std
        test_norm = (test_data - self.mean) / self.std
        print(f"  ...Computed unique mean and std for all {train_data.shape[1]} nodes.")

        print("➡️ Step 4: Creating (X, Y) samples for imputation...")
        # --- MODIFIED: Call new sample creation functions ---
        self.train_X, self.train_Y = self._create_training_samples(train_norm)
        self.val_X, self.val_Y = self._create_training_samples(val_norm)
        self.test_X, self.test_Y = self._create_testing_samples(test_norm)
        
        print(f"  ...Created {self.train_X.shape[0]} training samples.")
        print(f"  ...Created {self.val_X.shape[0]} validation samples.")
        print(f"  ...Created {self.test_X.shape[0]} testing/inference samples.")
        
        print(f"  ...Train_X shape: (Samples, Lags, HRRSM_Nodes) = {self.train_X.shape}")
        print(f"  ...Train_Y shape: (Samples, SM_Nodes) = {self.train_Y.shape}")
        
        print(f"  ...Test_X shape: (Samples, Lags, HRRSM_Nodes) = {self.test_X.shape}")
        print(f"  ...Test_Y shape: (Samples, SM_Nodes) = {self.test_Y.shape}")

        # --- Step 5 (Original) REMOVED ---
        # We no longer need np.expand_dims. The new model (in Models.py)
        # will need to be changed to accept this new input shape.
        print(f"➡️ Step 5: Skipping final 'expand_dims' (not needed for new task).")


    def _create_training_samples(self, data):
        """
        Creates (X, Y) samples for the TRAINING/VALIDATION sets.
        
        - X = HRRSM node data [t-lags+1 ... t]
        - Y = SM node data [t]
        
        This only creates a sample if the Y data (SM data) is NOT NaN.
        """
        X, Y = [], []
        
        # Start from the first point where we have a full 'lag' window
        for t in range(self.lags - 1, len(data)):
            
            # Check if the Y data (SM data) is available at this time.
            # We just check the first SM node. If it's NaN, we skip.
            y_check_val = data[t, self.sm_indices[0]]
            
            if np.isnan(y_check_val):
                # This is a "missing" row (e.g., t=1, 2, ... 14).
                # We can't train on it, so we skip it.
                continue
                
            # --- If we are here, t is a valid SM timestamp (e.g., 0, 15, 30...) ---
            
            # 1. Get the Y sample (the SM data at time t)
            y_sample = data[t, self.sm_indices]
            
            # 2. Get the X sample (the HRRSM data for the last 'lags' steps)
            input_start = t - (self.lags - 1)
            input_end = t + 1  # (Slice is exclusive, so t+1 includes t)
            x_sample = data[input_start:input_end, self.hrrsm_indices]
            
            # 3. Final check: Ensure the HRRSM data (input) has no NaNs
            if not np.isnan(x_sample).any():
                X.append(x_sample)
                Y.append(y_sample)
                
        return np.array(X), np.array(Y)

    def _create_testing_samples(self, data):
        """
        Creates (X, Y) samples for the TESTING/INFERENCE set.
        
        - X = HRRSM node data [t-lags+1 ... t]
        - Y = SM node data [t] (will be NaN most of the time)
        
        This creates a sample for *every* minute to fill in the blanks.
        """
        X, Y = [], []
        
        # Start from the first point where we have a full 'lag' window
        for t in range(self.lags - 1, len(data)):
            
            # 1. Get the Y sample (SM data at t)
            #    This is *expected* to be NaN for most 't'
            y_sample = data[t, self.sm_indices]
            
            # 2. Get the X sample (HRRSM data for the last 'lags' steps)
            input_start = t - (self.lags - 1)
            input_end = t + 1
            x_sample = data[input_start:input_end, self.hrrsm_indices]
            
            # 3. Final check: We can only make a prediction if
            #    the HRRSM input data is complete.
            if not np.isnan(x_sample).any():
                X.append(x_sample)
                Y.append(y_sample)
                
        return np.array(X), np.array(Y)


    def get_dataloaders(self, batch_size: int = 32, shuffle: bool = True):
        """
        Creates and returns PyTorch DataLoader objects.
        The return signature is kept the same as the original file.
        """
        print("➡️ Step 6: Creating PyTorch DataLoaders...")
        
        # --- Create Tensors ---
        # Note: The shapes are different from the original file.
        # train_X is (samples, lags, num_hrrsm_nodes)
        # train_Y is (samples, num_sm_nodes)
        train_dataset = TensorDataset(torch.from_numpy(self.train_X).float(), torch.from_numpy(self.train_Y).float())
        val_dataset = TensorDataset(torch.from_numpy(self.val_X).float(), torch.from_numpy(self.val_Y).float())
        
        # test_Y contains NaNs, which is correct.
        test_dataset = TensorDataset(torch.from_numpy(self.test_X).float(), torch.from_numpy(self.test_Y).float())

        # --- Create DataLoaders ---
        # We shuffle training data, but not validation or testing
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False) # Keep all test samples

        print("✅ All data processed and batched successfully.")
        
        # --- Return the same variables as the original file ---
        adj_matrix_t = torch.from_numpy(self.adj_matrix).float()
        mean_t = torch.tensor(self.mean).float()
        std_t = torch.tensor(self.std).float()

        return train_loader, val_loader, test_loader, adj_matrix_t, mean_t, std_t