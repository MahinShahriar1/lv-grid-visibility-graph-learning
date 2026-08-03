import scipy.io
import pandas as pd
import numpy as np

# --- Execution Constants ---
LV_NODES_FILE = 'LV_number_connected_to_MV.mat'
LOAD_SETTING_FILE = 'loadsetting.mat'
LINESETTING_FILE = 'Linesetting.mat'


def load_data():
    """Load all necessary MATLAB files."""
    try:
        mat_lv_nodes = scipy.io.loadmat(LV_NODES_FILE)['LV_number_connected_to_MV']
        mat_linesetting = scipy.io.loadmat(LINESETTING_FILE)['Linesetting']
        mat_load_setting = scipy.io.loadmat(LOAD_SETTING_FILE)['loadsetting']
        return mat_lv_nodes, mat_linesetting, mat_load_setting
    except Exception as e:
        print(f"Error loading MATLAB files: {e}")
        return None, None, None


def generate_split_matrices():
    """
    Generates three separate DataFrames: MV-LV, LV-LV, and Load/Phase Info.
    The MV-LV file now only contains the From/To bus IDs.
    The LV-LV file only contains topology (From_Bus, To_Bus, MV_Feeder_Bus).
    The Load/Phase Info is simplified to contain only the phase flags.
    """
    mat_lv_nodes, mat_linesetting, mat_load_setting = load_data()
    if mat_lv_nodes is None:
        return None, None, None

    mv_lv_links = []
    lv_lv_links = []
    load_info = []

    # Loop through the 21 feeders
    for i in range(mat_lv_nodes.shape[1]):
        mv_bus_id = int(mat_lv_nodes[0, i])  # MV Bus ID for this feeder
        lv_bus_list = [int(n) for n in mat_lv_nodes[1, i].flatten()]  # Absolute LV Bus IDs for this feeder
        lines_matrix = mat_linesetting[i, 0]  # N x 4 matrix for internal lines
        load_matrix = mat_load_setting[i, 0]  # N x 5 matrix for load data

        # --- 1. MV -> LV Link (21 rows total) ---
        lv_head_bus_id = lv_bus_list[0]
        # **Simplified to only include MV_From_Bus and LV_To_Bus**
        mv_lv_links.append({
            'MV_From_Bus': mv_bus_id,
            'LV_To_Bus': lv_head_bus_id
        })

        # --- 2. LV -> LV Links (404 rows total) ---
        for line_entry in lines_matrix:
            # Linesetting columns: [From_Rel, To_Rel, Length, Cable_Type_ID]
            from_bus_rel = int(line_entry[0])
            to_bus_rel = int(line_entry[1])

            try:
                # Convert Relative Index to Absolute Bus ID
                from_bus_abs = lv_bus_list[from_bus_rel - 1]
                to_bus_abs = lv_bus_list[to_bus_rel - 1]
            except IndexError:
                continue

            # Simplified to only include topology columns
            lv_lv_links.append({
                'From_Bus': from_bus_abs,
                'To_Bus': to_bus_abs,
                'MV_Feeder_Bus': mv_bus_id
            })

        # --- 3. Load/Phase Info (425 rows total) ---
        for load_entry in load_matrix:
            # Loadsetting columns: [Bus_Rel, P_A, P_B, P_C, Q_total]
            bus_index_rel = int(load_entry[0])
            p_a = load_entry[1]
            p_b = load_entry[2]
            p_c = load_entry[3]

            try:
                absolute_bus_id = lv_bus_list[bus_index_rel - 1]
            except IndexError:
                continue

            load_info.append({
                'Bus_ID': absolute_bus_id,
                'MV_Feeder_Bus': mv_bus_id,
                # Phase Flags (1/0)
                'Phase_A_Flag': 1 if p_a > 0 else 0,
                'Phase_B_Flag': 1 if p_b > 0 else 0,
                'Phase_C_Flag': 1 if p_c > 0 else 0,
            })

    # Convert to DataFrames
    df_mv_lv = pd.DataFrame(mv_lv_links)
    df_lv_lv = pd.DataFrame(lv_lv_links)
    df_load_info = pd.DataFrame(load_info)

    print(f"Generated MV-LV links: {len(df_mv_lv)} rows.")
    print(f"Generated LV-LV links: {len(df_lv_lv)} rows.")
    print(f"Generated Load/Phase Info: {len(df_load_info)} rows.")

    return df_mv_lv, df_lv_lv, df_load_info


# --- Execution ---
df_mv_lv, df_lv_lv, df_load_info = generate_split_matrices()

if df_mv_lv is not None:
    # Save and display MV-LV Connections (SIMPLIFIED)
    df_mv_lv.to_csv('mv_lv_connections.csv', index=False)
    print("\n--- 1. MV to LV Transformer Connections (21 rows - Topology Only) ---")
    print(df_mv_lv.head())

    # Save and display LV-LV Internal Lines (Topology Only)
    df_lv_lv.to_csv('lv_internal_lines.csv', index=False)
    print("\n--- 2. LV Internal Bus-to-Bus Lines (404 rows - Topology Only) ---")
    print(df_lv_lv.head())

    # Save and display Load and Phase Information (Simplified)
    df_load_info.to_csv('load_phase_info.csv', index=False)
    print("\n--- 3. Simplified Load and Phase Information by Bus (425 rows) ---")
    print(df_load_info.head())

else:
    print("\nFile generation failed.")
