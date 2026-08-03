import pandas as pd
import numpy as np
import sys
import time


def create_node_index_map(node_list):
    """
    Creates a dictionary mapping a Bus_ID to its index (row/col)
    in the new adjacency matrix.
    Example: {101: 0, 102: 1, 105: 2, ...}
    """
    return {node_id: index for index, node_id in enumerate(node_list)}


def build_full_adjacency_matrix(phase, all_data):
    """
    Builds a complete MV+LV adjacency matrix for a single phase.
    """

    # 1. Get all the data specific to this phase
    phase_flag_col = f'Phase_{phase}_Flag'
    total_node_list = all_data[f'nodes_mvlv_{phase.lower()}']
    old_mv_node_list = all_data[f'nodes_mv_{phase.lower()}']
    old_mv_conn_matrix = all_data[f'conn_mv_{phase.lower()}']

    # Get common data
    lv_lines = all_data['lv_lines']
    mv_lv_conn = all_data['mv_lv_conn']
    lv_phase_lookup = all_data['lv_phase_lookup']

    # 2. Create the "blank map"
    n = len(total_node_list)
    print(f"  Creating blank matrix of size {n}x{n}...")
    full_matrix = np.zeros((n, n), dtype=int)

    # Create the mapping of Bus_ID -> new_index
    # This is the most important part!
    index_map = create_node_index_map(total_node_list)
    # Create a fast set for checking if a node exists on this phase
    total_node_set = set(total_node_list)

    # 3. Add MV-to-MV Connections
    print("  Adding MV-to-MV connections...")
    # We must map the *old* MV indices to their *new* indices
    for r_idx, bus1 in enumerate(old_mv_node_list):
        for c_idx, bus2 in enumerate(old_mv_node_list):
            # If a connection exists in the *old* matrix
            if old_mv_conn_matrix[r_idx, c_idx] == 1:
                # Find their indices in the *new* matrix
                new_r_idx = index_map[bus1]
                new_c_idx = index_map[bus2]
                full_matrix[new_r_idx, new_c_idx] = 1

    # 4. Add LV-to-LV Connections
    print("  Adding LV-to-LV connections...")
    for _, row in lv_lines.iterrows():
        bus1 = row['From_Bus']
        bus2 = row['To_Bus']

        # CRUCIAL CHECK: Only add the line if *both* nodes
        # are active on this specific phase.
        bus1_active = (bus1, phase_flag_col) in lv_phase_lookup
        bus2_active = (bus2, phase_flag_col) in lv_phase_lookup

        if bus1_active and bus2_active:
            # Find their indices in the *new* matrix
            new_r_idx = index_map[bus1]
            new_c_idx = index_map[bus2]
            # Add the connection (symmetrically)
            full_matrix[new_r_idx, new_c_idx] = 1
            full_matrix[new_c_idx, new_r_idx] = 1

    # 5. Add MV-to-LV (Transformer) Connections
    print("  Adding MV-to-LV (transformer) connections...")
    for _, row in mv_lv_conn.iterrows():
        bus_mv = row['MV_From_Bus']
        bus_lv = row['LV_To_Bus']

        # CRUCIAL CHECK: Only add the transformer if *both* of its
        # nodes are part of this phase's complete node list.
        if bus_mv in total_node_set and bus_lv in total_node_set:
            # Find their indices in the *new* matrix
            new_r_idx = index_map[bus_mv]
            new_c_idx = index_map[bus_lv]
            # Add the connection (symmetrically)
            full_matrix[new_r_idx, new_c_idx] = 1
            full_matrix[new_c_idx, new_r_idx] = 1

    return full_matrix


def main():
    print("Starting network topology build...")
    start_time = time.time()

    try:
        # This will hold all our loaded data
        all_data = {}

        # --- 1. Load all files ---
        print("Loading all 11 input files...")

        # Load Node Lists (as flat Python lists)
        all_data['nodes_mv_a'] = pd.read_csv("nodes_phaseA.csv", header=None).iloc[0].tolist()
        all_data['nodes_mv_b'] = pd.read_csv("nodes_phaseB.csv", header=None).iloc[0].tolist()
        all_data['nodes_mv_c'] = pd.read_csv("nodes_phaseC.csv", header=None).iloc[0].tolist()

        all_data['nodes_mvlv_a'] = pd.read_csv("nodesMVLV_phaseA.csv", header=None).iloc[0].tolist()
        all_data['nodes_mvlv_b'] = pd.read_csv("nodesMVLV_phaseB.csv", header=None).iloc[0].tolist()
        all_data['nodes_mvlv_c'] = pd.read_csv("nodesMVLV_phaseC.csv", header=None).iloc[0].tolist()

        # Load MV Connection Matrices (as numpy arrays)
        all_data['conn_mv_a'] = pd.read_csv("Conn_phaseA.csv", header=None).values
        all_data['conn_mv_b'] = pd.read_csv("Conn_phaseB.csv", header=None).values
        all_data['conn_mv_c'] = pd.read_csv("Conn_phaseC.csv", header=None).values

        # Load Edge Lists (as DataFrames)
        all_data['lv_lines'] = pd.read_csv("lv_internal_lines.csv")
        all_data['mv_lv_conn'] = pd.read_csv("mv_lv_connections.csv")

        # Load LV Phase Info and create a fast lookup set
        df_lv_info = pd.read_csv("load_phase_info.csv")
        lv_phase_lookup = set()
        for _, row in df_lv_info.iterrows():
            if row['Phase_A_Flag'] == 1:
                lv_phase_lookup.add((row['Bus_ID'], 'Phase_A_Flag'))
            if row['Phase_B_Flag'] == 1:
                lv_phase_lookup.add((row['Bus_ID'], 'Phase_B_Flag'))
            if row['Phase_C_Flag'] == 1:
                lv_phase_lookup.add((row['Bus_ID'], 'Phase_C_Flag'))
        all_data['lv_phase_lookup'] = lv_phase_lookup

        print("All files loaded and pre-processed.")

        # --- 2. Process Phase A ---
        print("\nProcessing Phase A...")
        matrix_A = build_full_adjacency_matrix('A', all_data)
        print(f"  Saving ConnMVLV_phaseA.csv with shape {matrix_A.shape}...")
        pd.DataFrame(matrix_A).to_csv("ConnMVLV_phaseA.csv", header=False, index=False)
        print("  Phase A complete.")

        # --- 3. Process Phase B ---
        print("\nProcessing Phase B...")
        matrix_B = build_full_adjacency_matrix('B', all_data)
        print(f"  Saving ConnMVLV_phaseB.csv with shape {matrix_B.shape}...")
        pd.DataFrame(matrix_B).to_csv("ConnMVLV_phaseB.csv", header=False, index=False)
        print("  Phase B complete.")

        # --- 4. Process Phase C ---
        print("\nProcessing Phase C...")
        matrix_C = build_full_adjacency_matrix('C', all_data)
        print(f"  Saving ConnMVLV_phaseC.csv with shape {matrix_C.shape}...")
        pd.DataFrame(matrix_C).to_csv("ConnMVLV_phaseC.csv", header=False, index=False)
        print("  Phase C complete.")

        # --- 5. Done ---
        end_time = time.time()
        print(f"\nSuccessfully built all 3 matrices in {end_time - start_time:.2f} seconds.")

    except FileNotFoundError as e:
        print(f"\n--- ERROR ---")
        print(f"Could not find a required file: {e.filename}")
        print("Please ensure all 11 data files are available in the same directory.")
        sys.exit(1)
    except KeyError as e:
        print(f"\n--- ERROR ---")
        print(f"A data key was missing: {e}")
        print("This may be due to a malformed file or a script error.")
        sys.exit(1)
    except Exception as e:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED ---")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()
