import pandas as pd
import os


def generate_lv_node_lists(input_filename='load_phase_info.csv'):
    """
    Reads the load phase information, extracts the Low Voltage (LV) bus IDs
    for each phase (A, B, C) where a load is present (flag = 1), and saves
    each list to a new CSV file in a single-row format.
    """
    try:
        # Load the dataset
        df = pd.read_csv(input_filename)
    except FileNotFoundError:
        print(f"Error: Input file '{input_filename}' not found.")
        return
    except Exception as e:
        print(f"Error reading file '{input_filename}': {e}")
        return

    # 1. Extract nodes for each phase
    # Filter for rows where the respective phase flag is 1, and get the Bus_ID

    # Phase A nodes
    nodes_a = df[df['Phase_A_Flag'] == 1]['Bus_ID'].sort_values().tolist()

    # Phase B nodes
    nodes_b = df[df['Phase_B_Flag'] == 1]['Bus_ID'].sort_values().tolist()

    # Phase C nodes
    nodes_c = df[df['Phase_C_Flag'] == 1]['Bus_ID'].sort_values().tolist()

    # Define the output files and corresponding node lists
    output_files = {
        'nodesLV_phaseA.csv': nodes_a,
        'nodesLV_phaseB.csv': nodes_b,
        'nodesLV_phaseC.csv': nodes_c
    }

    # 2. Save each node list to a new CSV file
    print("--- Generating LV Node Files ---")
    for filename, nodes in output_files.items():
        # Create a single-row DataFrame for the output
        # Transpose the list of nodes to be a single row, and remove the header/index
        output_df = pd.DataFrame([nodes])

        # Save to CSV
        output_df.to_csv(filename, index=False, header=False)
        print(f"Generated {filename} with {len(nodes)} nodes.")

    print("\nGeneration complete.")


# Execute the function
generate_lv_node_lists()
