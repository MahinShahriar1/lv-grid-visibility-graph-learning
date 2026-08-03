import pandas as pd
import numpy as np
import os

# --- 0. Define the output directory ---
SAVE_PATH = './processed_data/'
os.makedirs(SAVE_PATH, exist_ok=True)
print(f"All processed files will be saved in: '{SAVE_PATH}'")
print("-" * 50)

# --- 1. Load All Raw Data Files ---
print("Step 1: Loading raw data files...")
try:
    # The file with the list of all 30 node IDs
    nodes_df = pd.read_csv('nodes_phaseA.csv', header=None)
    full_node_list = nodes_df.iloc[0].tolist()

    # The 30x30 adjacency matrix
    adj_matrix_30x30 = pd.read_csv('Conn_phaseA.csv', header=None).values

    # The file with measurement data for 29 nodes
    voltage_df = pd.read_csv('MVtrue_Vmag_phaseA.csv', header=None)
    features_29_nodes = voltage_df.iloc[:, 1:].values

    print("All files loaded successfully.")
    print("-" * 50)

except FileNotFoundError as e:
    print(f"FATAL ERROR: Could not find '{e.filename}'. Please ensure all CSV files are in the same directory.")
    exit()

# --- 2. Filter Adjacency Matrix to Match Data ---
print("Step 2: Filtering adjacency matrix to 29x29...")
# We know the data file has 29 nodes, corresponding to the first 29 nodes in the list
nodes_with_data = full_node_list[:29]
missing_node = full_node_list[29]  # The 30th node in the list is the one missing
missing_node_index = 29

print(f"Node to remove: ID {missing_node} at index {missing_node_index}")

# Remove the row and column for the missing node
adj_matrix_filtered = np.delete(adj_matrix_30x30, missing_node_index, axis=0)
adj_matrix_filtered = np.delete(adj_matrix_filtered, missing_node_index, axis=1)

print(f"New adjacency matrix shape is: {adj_matrix_filtered.shape}")
print("-" * 50)

# --- 3. Split the Feature Data (70/10/20) ---
print("Step 3: Splitting feature data...")
n_timesteps = features_29_nodes.shape[0]
train_split = int(n_timesteps * 0.7)
val_split = int(n_timesteps * 0.8)

train_data = features_29_nodes[:train_split]
val_data = features_29_nodes[train_split:val_split]
test_data = features_29_nodes[val_split:]
print("Data split into training, validation, and test sets.")
print("-" * 50)

# --- 4. Z-Score Normalization ---
print("Step 4: Applying Z-Score Normalization...")
mean = train_data.mean()
std = train_data.std()
print(f"Calculated Mean: {mean:.4f}, Std Dev: {std:.4f} (from training data only)")

train_data_normalized = (train_data - mean) / std
val_data_normalized = (val_data - mean) / std
test_data_normalized = (test_data - mean) / std
print("Normalization applied to all sets.")
print("-" * 50)

# --- 5. Create Sliding Window Samples ---
print("Step 5: Creating sliding window samples (X, Y pairs)...")


def create_samples(data, input_window, output_window):
    X, Y = [], []
    for i in range(len(data) - input_window - output_window + 1):
        X.append(data[i:(i + input_window)])
        Y.append(data[(i + input_window):(i + input_window + output_window)])
    return np.array(X), np.array(Y)


INPUT_WINDOW = 60
OUTPUT_WINDOW = 15

train_X, train_Y = create_samples(train_data_normalized, INPUT_WINDOW, OUTPUT_WINDOW)
val_X, val_Y = create_samples(val_data_normalized, INPUT_WINDOW, OUTPUT_WINDOW)
test_X, test_Y = create_samples(test_data_normalized, INPUT_WINDOW, OUTPUT_WINDOW)
print("Sliding window samples created.")
print(f"Shape of training features (train_X): {train_X.shape}")
print("-" * 50)

# --- 6. Save All Processed Files ---
print(f"Step 6: Saving all processed arrays to '{SAVE_PATH}'...")
np.save(os.path.join(SAVE_PATH, 'train_X.npy'), train_X)
np.save(os.path.join(SAVE_PATH, 'train_Y.npy'), train_Y)
np.save(os.path.join(SAVE_PATH, 'val_X.npy'), val_X)
np.save(os.path.join(SAVE_PATH, 'val_Y.npy'), val_Y)
np.save(os.path.join(SAVE_PATH, 'test_X.npy'), test_X)
np.save(os.path.join(SAVE_PATH, 'test_Y.npy'), test_Y)
# Save the new, filtered 29x29 adjacency matrix
np.save(os.path.join(SAVE_PATH, 'adj_matrix.npy'), adj_matrix_filtered)

print("\n✅ All data has been processed and saved successfully!")