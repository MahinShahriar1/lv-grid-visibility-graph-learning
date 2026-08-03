import pandas as pd
import numpy as np
import scipy.io
import sys
import time


def load_node_list(filename):
    """Loads a node list CSV (from a single row) as a flat list of strings."""
    try:
        return [str(node) for node in pd.read_csv(filename, header=None).iloc[0].tolist()]
    except FileNotFoundError:
        print(f"\n--- FATAL ERROR ---")
        print(f"Could not find required file: {filename}")
        print("Please make sure all node list files are in the same directory.")
        sys.exit(1)
    except Exception as e:
        print(f"\n--- FATAL ERROR ---")
        print(f"Error loading {filename}: {e}")
        sys.exit(1)


def main():
    print("Starting voltage extraction and validation process...")
    start_time = time.time()

    try:
        # --- 1. Load All Input Files ---
        print("Loading all input files...")

        # Load the main voltage data
        try:
            mat_data = scipy.io.loadmat('dataV1.mat')
            data_key = [k for k in mat_data.keys() if not k.startswith('__')][0]
            # dataV1 is a (100, 1377) matrix (or larger)
            dataV1 = mat_data[data_key]

            # --- FIX: Remove trailing zero rows ---
            # Finds the last row that has non-zero data and keeps everything before it
            any_data_rows = np.any(dataV1, axis=1)
            last_data_row = np.where(any_data_rows)[0][-1]
            dataV1 = dataV1[:last_data_row + 1, :]
            # --- End Fix ---

        except FileNotFoundError:
            print("\n--- FATAL ERROR ---")
            print("Could not find required file: matlab.mat")
            sys.exit(1)

        print(f"  Loaded voltage data '{data_key}' and trimmed to shape {dataV1.shape}")

        # Load the node lists that define which nodes are active on each phase
        nodes_a_list = load_node_list('nodesMVLV_phaseA.csv')
        nodes_b_list = load_node_list('nodesMVLV_phaseB.csv')
        nodes_c_list = load_node_list('nodesMVLV_phaseC.csv')

        # Create fast lookup sets
        active_nodes_A = set(nodes_a_list)
        active_nodes_B = set(nodes_b_list)
        active_nodes_C = set(nodes_c_list)

        print("  Loaded all node lists and created lookup sets.")

        # --- 2. Create Master Node List & Run FATAL Validation Check ---

        # Combine all node lists to get a set of unique nodes
        all_nodes_set = set(nodes_a_list) | set(nodes_b_list) | set(nodes_c_list)

        # -----------------------------------------------------------------
        # *** THIS IS THE CRITICAL FIX ***
        # -----------------------------------------------------------------
        # We sort the nodes NUMERICALLY, not alphabetically.
        # This ensures ['1', '2', '3', ... '10', '11'] order.
        print("  Creating numerically-sorted master node list...")
        try:
            # 1. Convert all node IDs (strings) to integers
            all_nodes_int = [int(node) for node in all_nodes_set]
        except ValueError:
            print("\n--- FATAL ERROR ---")
            print("Your node list files contain non-numeric node names (e.g., 'Node_A').")
            print("This script assumed nodes are numbers ('1', '2', '100') to sort correctly.")
            sys.exit(1)

        # 2. Sort the integers
        all_nodes_int_sorted = sorted(all_nodes_int)

        # 3. Convert back to strings to create the master list
        master_node_list = [str(node) for node in all_nodes_int_sorted]
        # -----------------------------------------------------------------
        # *** END FIX ***
        # -----------------------------------------------------------------

        print(f"  Validating data consistency...")
        if len(master_node_list) * 3 != dataV1.shape[1]:
            print(f"\n--- FATAL ERROR: DATA INCONSISTENCY ---")
            print(f"Node count mismatch! Your lists have {len(master_node_list)} unique nodes.")
            print(f"This requires {len(master_node_list) * 3} data columns.")
            print(f"But the 'matlab.mat' file has {dataV1.shape[1]} columns.")
            print("The node lists do not match the data file. Stopping.")
            sys.exit(1)

        print(f"  [SUCCESS] Data is consistent: {len(master_node_list)} nodes * 3 = {dataV1.shape[1]} columns.")

        # --- 3. Iterate, Extract, and Run In-Depth Validation ---
        print("Extracting and de-interlacing phase data...")

        # Use dictionaries to store data, linking node_id to its data column
        phase_A_data = {}
        phase_B_data = {}
        phase_C_data = {}

        # Validation counters
        unexpected_voltage_count = 0
        missing_voltage_count = 0

        # Loop through the nodes in the MASTER (numerical) list
        for i, node_id in enumerate(master_node_list):
            # Get the (100,) data columns for this node
            # This now correctly maps index `i` to `node_id`
            # e.g., i=2, node_id='3' -> grabs cols 6, 7, 8
            col_A = dataV1[:, i * 3]
            col_B = dataV1[:, i * 3 + 1]
            col_C = dataV1[:, i * 3 + 2]

            # --- Phase A Check ---
            if node_id in active_nodes_A:
                phase_A_data[node_id] = col_A  # Store data by node_id
                if np.all(col_A == 0):
                    missing_voltage_count += 1
                    print(f"  [VALIDATION WARNING] Node {node_id} is on Phase A but has all-zero voltage data.")
            else:
                if np.any(col_A != 0):
                    unexpected_voltage_count += 1
                    print(f"  [VALIDATION ERROR] Node {node_id} is NOT on Phase A, but has non-zero voltage data.")

            # --- Phase B Check ---
            if node_id in active_nodes_B:
                phase_B_data[node_id] = col_B
                if np.all(col_B == 0):
                    missing_voltage_count += 1
                    print(f"  [VALIDATION WARNING] Node {node_id} is on Phase B but has all-zero voltage data.")
            else:
                if np.any(col_B != 0):
                    unexpected_voltage_count += 1
                    print(f"  [VALIDATION ERROR] Node {node_id} is NOT on Phase B, but has non-zero voltage data.")

            # --- Phase C Check ---
            if node_id in active_nodes_C:
                phase_C_data[node_id] = col_C
                if np.all(col_C == 0):
                    missing_voltage_count += 1
                    print(f"  [VALIDATION WARNING] Node {node_id} is on Phase C but has all-zero voltage data.")
            else:
                if np.any(col_C != 0):
                    unexpected_voltage_count += 1
                    print(f"  [VALIDATION ERROR] Node {node_id} is NOT on Phase C, but has non-zero voltage data.")

        print("  Data extraction complete.")

        # --- 4. Report Validation Summary ---
        print("\n--- Validation Summary ---")
        if unexpected_voltage_count == 0:
            print("  [SUCCESS] No unexpected voltages found. Data is consistent.")
        else:
            print(
                f"  [!!! ERROR !!!] Found {unexpected_voltage_count} instances of unexpected voltage on inactive phases.")

        if missing_voltage_count == 0:
            print("  [INFO] All active nodes have at least one non-zero voltage value.")
        else:
            print(f"  [WARNING] Found {missing_voltage_count} instances of active nodes with all-zero voltage.")
        print("----------------------------\n")

        # --- 5. Build and Save DataFrames ---
        print("Building and saving new CSV files...")

        # Create DataFrames from the dictionaries.
        # This correctly maps "node 3" to its REAL data.
        df_A_all = pd.DataFrame(phase_A_data)
        df_B_all = pd.DataFrame(phase_B_data)
        df_C_all = pd.DataFrame(phase_C_data)

        # Now, re-order the columns to match your original phase CSV files
        # This works because df_A_all now contains the *correct* data for every node.
        try:
            df_A = df_A_all[nodes_a_list]
            df_B = df_B_all[nodes_b_list]
            df_C = df_C_all[nodes_c_list]
        except KeyError as e:
            print(f"\n--- FATAL ERROR: KEY ERROR ---")
            print(f"A node in your phase CSV list was not found in the extracted data.")
            print(f"This should not happen. Missing node: {e}")
            sys.exit(1)

        # Save the new, CORRECTLY-ORDERED and CORRECTLY-MAPPED DataFrames
        df_A.to_csv("MVLV_VmagTure_phaseA.csv", index=False, header=False)
        print(f"  Saved MVLV_VmagTure_phaseA.csv with shape {df_A.shape}.")

        df_B.to_csv("MVLV_VmagTure_phaseB.csv", index=False, header=False)
        print(f"  Saved MVLV_VmagTure_phaseB.csv with shape {df_B.shape}.")

        df_C.to_csv("MVLV_VmagTure_phaseC.csv", index=False, header=False)
        print(f"  Saved MVLV_VmagTure_phaseC.csv with shape {df_C.shape}.")

        end_time = time.time()
        print(f"\nSuccessfully extracted and validated all voltage data in {end_time - start_time:.2f} seconds.")

    except Exception as e:
        print(f"\n--- AN UNEXPECTED ERROR OCCURRED ---")
        print(e)
        sys.exit(1)


if __name__ == "__main__":
    main()