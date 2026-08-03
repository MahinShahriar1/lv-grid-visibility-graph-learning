import pandas as pd
import numpy as np

try:
    # --- 1. Load All Input Files ---

    # Load the LV phase info
    df_lv = pd.read_csv("load_phase_info.csv")

    # Load the MV node lists.
    mv_nodes_a = pd.read_csv("nodes_phaseA.csv", header=None).iloc[0].tolist()
    mv_nodes_b = pd.read_csv("nodes_phaseB.csv", header=None).iloc[0].tolist()
    mv_nodes_c = pd.read_csv("nodes_phaseC.csv", header=None).iloc[0].tolist()

    print("--- Initial Data Loaded ---")
    print(f"Loaded {len(df_lv)} LV node entries.")
    print(f"Loaded {len(mv_nodes_a)} MV nodes for Phase A.")
    print(f"Loaded {len(mv_nodes_b)} MV nodes for Phase B.")
    print(f"Loaded {len(mv_nodes_c)} MV nodes for Phase C.")

    # --- 1b. Calculate Unique MV Nodes ---
    unique_mv_nodes = set(mv_nodes_a) | set(mv_nodes_b) | set(mv_nodes_c)
    unique_mv_count = len(unique_mv_nodes)
    print(f"Found {unique_mv_count} unique MV nodes.")

    # --- 2. Extract LV Nodes from the DataFrame ---

    # Filter the DataFrame where the flag is 1, then select the 'Bus_ID' column
    lv_nodes_a = df_lv[df_lv['Phase_A_Flag'] == 1]['Bus_ID'].tolist()
    lv_nodes_b = df_lv[df_lv['Phase_B_Flag'] == 1]['Bus_ID'].tolist()
    lv_nodes_c = df_lv[df_lv['Phase_C_Flag'] == 1]['Bus_ID'].tolist()

    print(f"\nExtracted {len(lv_nodes_a)} LV nodes for Phase A.")
    print(f"Extracted {len(lv_nodes_b)} LV nodes for Phase B.")
    print(f"Extracted {len(lv_nodes_c)} LV nodes for Phase C.")

    # --- 2b. Calculate Unique LV Nodes ---
    unique_lv_nodes = set(lv_nodes_a) | set(lv_nodes_b) | set(lv_nodes_c)
    unique_lv_count = len(unique_lv_nodes)
    print(f"Found {unique_lv_count} unique LV nodes.")

    # --- 3. Combine MV and LV Node Lists ---

    # Use sets to automatically handle duplicates when combining
    # Then convert back to a list and sort it
    nodesMVLV_phaseA = sorted(list(set(mv_nodes_a) | set(lv_nodes_a)))
    nodesMVLV_phaseB = sorted(list(set(mv_nodes_b) | set(lv_nodes_b)))
    nodesMVLV_phaseC = sorted(list(set(mv_nodes_c) | set(lv_nodes_c)))

    # --- 4. Save New Node Lists to CSV ---

    # Create new DataFrames with a single row to match the original format
    df_a = pd.DataFrame([nodesMVLV_phaseA])
    df_b = pd.DataFrame([nodesMVLV_phaseB])
    df_c = pd.DataFrame([nodesMVLV_phaseC])

    # Save to new CSV files without headers or an index column
    df_a.to_csv("nodesMVLV_phaseA.csv", header=False, index=False)
    df_b.to_csv("nodesMVLV_phaseB.csv", header=False, index=False)
    df_c.to_csv("nodesMVLV_phaseC.csv", header=False, index=False)

    print("""
Successfully saved:
    * nodesMVLV_phaseA.csv
    * nodesMVLV_phaseB.csv
    * nodesMVLV_phaseC.csv""")

    # --- 5. Count Per-Phase and Total Unique Nodes ---

    count_a = len(nodesMVLV_phaseA)
    count_b = len(nodesMVLV_phaseB)
    count_c = len(nodesMVLV_phaseC)

    # Calculate total unique nodes by combining the unique MV and LV sets
    total_unique_nodes = unique_mv_nodes | unique_lv_nodes
    total_unique_count = len(total_unique_nodes)

    # --- 6. Print Final Counts ---

    print("\n--- Summary of All Node Counts ---")

    print(f"\nTotal Unique MV Nodes: {unique_mv_count}")
    print(f"Total Unique LV Nodes: {unique_lv_count}")
    print(f"Total Unique Nodes (MV + LV): {total_unique_count}")

    print("\n--- Combined (MV+LV) Per-Phase Counts ---")
    print(f"Total nodes in nodesMVLV_phaseA.csv: {count_a}")
    print(f"Total nodes in nodesMVLV_phaseB.csv: {count_b}")
    print(f"Total nodes in nodesMVLV_phaseC.csv: {count_c}")


except FileNotFoundError as e:
    print(f"Error: Could not find a file. Make sure all files are present.")
    print(f"Details: {e}")
except Exception as e:
    print(f"An error occurred during processing: {e}")
