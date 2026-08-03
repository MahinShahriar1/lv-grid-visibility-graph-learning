import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import community.community_louvain as community_louvain
import cdlib.algorithms
import warnings
import os

# Suppress the FutureWarning from pandas/numpy operations
warnings.simplefilter(action='ignore', category=FutureWarning)


# =================================================================
# --- CORE UTILITY FUNCTIONS ---
# =================================================================

def load_all_unique_nodes():
    """Loads all three node lists and returns a sorted list of unique node names."""
    try:
        nodes_a = set(pd.read_csv('nodesMVLV_phaseA.csv', header=None).iloc[0].astype(str).tolist())
        nodes_b = set(pd.read_csv('nodesMVLV_phaseB.csv', header=None).iloc[0].astype(str).tolist())
        nodes_c = set(pd.read_csv('nodesMVLV_phaseC.csv', header=None).iloc[0].astype(str).tolist())
        all_unique_nodes = sorted(list(nodes_a.union(nodes_b).union(nodes_c)), key=int)
        return all_unique_nodes
    except Exception as e:
        print(f"Error loading node lists: {e}")
        return None


def load_hybrid_df(conn_file, corr_file, nodes_file):
    """
    Loads one phase of PHYSICAL connections AND data correlations,
    multiplies them (Conn * |Corr|), and returns the 'hybrid' DataFrame.
    """
    try:
        node_names = pd.read_csv(nodes_file, header=None).iloc[0].astype(str).tolist()
        df_conn = pd.read_csv(conn_file, header=None)
        df_conn.index = node_names
        df_conn.columns = node_names
        df_corr = pd.read_csv(corr_file, header=None)
        df_corr.index = node_names
        df_corr.columns = node_names

        df_hybrid = df_conn * np.abs(df_corr)
        # df_hybrid = np.abs(df_corr)
        np.fill_diagonal(df_hybrid.values, 0)
        return df_hybrid
    except Exception as e:
        print(f"Error processing files {conn_file}, {corr_file}: {e}")
        return None


def combine_matrices(df_a, df_b, df_c):
    """
    Aligns and calculates the element-wise average of three hybrid matrices
    to create the Combined baseline matrix.
    """
    all_nodes = load_all_unique_nodes()
    df_a = df_a.reindex(index=all_nodes, columns=all_nodes).fillna(0)
    df_b = df_b.reindex(index=all_nodes, columns=all_nodes).fillna(0)
    df_c = df_c.reindex(index=all_nodes, columns=all_nodes).fillna(0)
    df_combined = (df_a.add(df_b).add(df_c)) / 3.0
    # df_combined = (df_a.add(df_b).add(df_c))
    np.fill_diagonal(df_combined.values, 0)
    return df_combined


def invert_correlation_weights(x):
    """Inverts a hybrid weight score to represent cost for community detection."""
    if x <= 0:
        return 0
    else:
        return 1.0 / x


def get_zone_captains_with_scores(G_main_int, inv_mapping, zones, algorithm_name, hybrid_matrix_filename):
    """
    Analyzes zones and finds the 'captain' using the local hybrid matrix.
    Rule 1: Highest Total Intra-Zone HYBRID Score.
    Rule 2: (Tie-breaker) Highest Betweenness Centrality (using inverse cost).
    Returns: (sorted_list_of_captains, zone_to_captain_dict)
    """
    print("\n" + "=" * 50)
    print(f" Finding 'Captains' for {algorithm_name} Algorithm")
    print(f"   (Captain = Node with Highest Total HYBRID Score in Zone)")
    print("=" * 50)

    captain_list = []
    zone_to_captain = {}
    df_hybrid_local = pd.read_csv(hybrid_matrix_filename, index_col=0)
    df_hybrid_local.columns = df_hybrid_local.columns.map(str)
    df_hybrid_local.index = df_hybrid_local.index.map(str)

    for zone_id, node_list_int in zones.items():
        # convert integer-labeled nodes back to string labels using inv_mapping
        node_list_str = sorted([inv_mapping[n] for n in node_list_int], key=lambda x: int(x))

        if len(node_list_str) == 1:
            captain_node = node_list_str[0]
        else:
            zone_matrix = df_hybrid_local.loc[node_list_str, node_list_str]
            local_centrality_score = zone_matrix.sum(axis=1)
            sorted_scores = local_centrality_score.sort_values(ascending=False)
            captain_score = sorted_scores.iloc[0]
            tied_nodes = sorted_scores[sorted_scores == captain_score].index.tolist()

            if len(tied_nodes) == 1:
                captain_node = tied_nodes[0]
            else:
                # tie-breaker by betweenness centrality on the cost graph
                subgraph_for_centrality = G_main_int.subgraph(node_list_int)
                zone_betweenness = nx.betweenness_centrality(
                    subgraph_for_centrality,
                    weight='weight',
                    normalized=False
                )
                # map scores back to string names
                zone_betweenness_str = {inv_mapping[node_int]: score for node_int, score in zone_betweenness.items()}
                tied_node_scores_bc = {node: zone_betweenness_str.get(node, 0) for node in tied_nodes}
                captain_node = max(tied_node_scores_bc, key=tied_node_scores_bc.get)

            # print detailed zone scores
            print(f"\n--- Zone {zone_id} ({algorithm_name}) ---")
            for node, score in sorted_scores.items():
                marker = "->" if node == captain_node else "  "
                print(f"    {marker} Node {node}:\t{score:.4f}")
            print(f"  Captain: Node {captain_node} (Score: {captain_score:.4f})")

        captain_list.append(captain_node)
        zone_to_captain[zone_id] = captain_node

    # return sorted captain list for compatibility and dict
    return sorted(captain_list, key=lambda x: int(x)), zone_to_captain


# =================================================================
# --- VISUALIZATION FUNCTIONS ---
# =================================================================

def draw_community_graph(G, pos, partition, captains, algorithm_name, phase_name):
    """Draws the community graph with captains highlighted."""
    print(f"Drawing {algorithm_name} network graph for {phase_name}...")
    plt.figure(figsize=(12, 12))
    num_zones = len(set(partition.values())) if partition else 1
    cmap = plt.cm.get_cmap('tab20', num_zones)
    colors = [cmap(partition[node]) if node in partition else (0.5, 0.5, 0.5, 1.0) for node in G.nodes()]
    sizes = [1500 if node in captains else 400 for node in G.nodes()]

    weights = [G.edges[u, v].get('weight', 0) for u, v in G.edges()]
    corr_weights = [((1 / w) * 0.5) if w > 0 else 0.1 for w in weights]

    nx.draw_networkx_edges(G, pos, alpha=0.2, width=corr_weights, edge_color='gray')
    nx.draw_networkx_nodes(G, pos, node_color=colors, node_size=sizes, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_labels(G, pos, font_size=9)
    plt.title(f'Network Communities ({algorithm_name}) - {phase_name}', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    safe_name = phase_name.replace(" ", "_")
    plt.savefig(f'community_plot_{safe_name}_{algorithm_name.lower()}.png', dpi=300)
    plt.close()


def draw_captain_justification_charts(zones, algorithm_name, phase_name, hybrid_matrix_filename, inv_mapping,
                                      G_weighted_int):
    """Generates a separate bar chart for each zone to show why the captain was chosen."""
    print(f"Drawing {algorithm_name} captain justification charts for {phase_name}...")
    if not zones:
        print("  ...no zones to draw.")
        return

    df_hybrid_local = pd.read_csv(hybrid_matrix_filename, index_col=0)
    df_hybrid_local.columns = df_hybrid_local.columns.map(str)
    df_hybrid_local.index = df_hybrid_local.index.map(str)

    num_zones = len(zones)
    cols = int(np.ceil(np.sqrt(num_zones)))
    rows = int(np.ceil(num_zones / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
    axes = axes.flatten()

    for i, (zone_id, node_list_int) in enumerate(zones.items()):
        ax = axes[i]
        node_list_str = sorted([inv_mapping[n] for n in node_list_int], key=lambda x: int(x))

        if len(node_list_str) == 1:
            ax.bar(node_list_str[0], 0)
            captain_node = node_list_str[0]
        else:
            zone_matrix = df_hybrid_local.loc[node_list_str, node_list_str]
            local_centrality_score = zone_matrix.sum(axis=1)
            sorted_scores = local_centrality_score.sort_values(ascending=False)
            captain_score = sorted_scores.iloc[0]
            tied_nodes = sorted_scores[sorted_scores == captain_score].index.tolist()

            if len(tied_nodes) == 1:
                captain_node = tied_nodes[0]
            else:
                subgraph_for_centrality = G_weighted_int.subgraph(node_list_int)
                zone_betweenness = nx.betweenness_centrality(subgraph_for_centrality, weight='weight', normalized=False)
                zone_betweenness_str = {inv_mapping[node_int]: score for node_int, score in zone_betweenness.items()}
                tied_node_scores_bc = {node: zone_betweenness_str.get(node, 0) for node in tied_nodes}
                captain_node = max(tied_node_scores_bc, key=tied_node_scores_bc.get)

            sorted_scores.plot(kind='bar', ax=ax, color='gray')
            captain_bar_index = sorted_scores.index.tolist().index(captain_node)
            ax.patches[captain_bar_index].set_facecolor('red')

        ax.set_title(f"Zone {zone_id} (Captain: {captain_node})", fontsize=9)
        ax.set_ylabel("Intra-Zone Score", fontsize=8)
        ax.tick_params(axis='x', rotation=90, labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'Captain Justification ({algorithm_name}) - {phase_name}', fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    safe_name = phase_name.replace(" ", "_")
    plt.savefig(f'captain_justification_{safe_name}_{algorithm_name.lower()}.png')
    plt.close()

def export_zone_csvs(
    phase_name: str,
    algorithm_name: str,
    partition_str: dict,
    zones_str: dict,
    zone_to_captain: dict,
    hybrid_matrix_filename: str,
    out_dir: str = "community_outputs",
):
    """
    Exports:
      (1) memberships CSV: one row per node with its zone + captain flag
      (2) zone-scores CSV: long-form scores used in captain justification plots
    """

    os.makedirs(out_dir, exist_ok=True)

    # Normalize phase string for filenames
    safe_phase = phase_name.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
    safe_algo = algorithm_name.lower()

    # --- (A) Node -> Zone membership table ---
    rows_membership = []
    for node_str, zid in partition_str.items():
        captain_node = zone_to_captain.get(zid, None)
        rows_membership.append({
            "phase": phase_name,
            "algorithm": algorithm_name,
            "node": str(node_str),
            "zone_id": int(zid) if isinstance(zid, (int, np.integer)) else zid,
            "captain_node": str(captain_node) if captain_node is not None else "",
            "is_captain": int(str(node_str) == str(captain_node)) if captain_node is not None else 0,
        })

    df_membership = pd.DataFrame(rows_membership).sort_values(
        by=["zone_id", "node"],
        key=lambda s: s.map(lambda x: int(x) if str(x).isdigit() else str(x))
    )
    df_membership.to_csv(os.path.join(out_dir, f"memberships_{safe_phase}_{safe_algo}.csv"), index=False)

    # --- (B) Per-zone intra-zone scores (same as your plot logic) ---
    df_hybrid_local = pd.read_csv(hybrid_matrix_filename, index_col=0)
    df_hybrid_local.columns = df_hybrid_local.columns.map(str)
    df_hybrid_local.index = df_hybrid_local.index.map(str)

    rows_scores = []
    for zid, members in zones_str.items():
        members = [str(m) for m in members]
        captain_node = zone_to_captain.get(zid, None)

        if len(members) == 1:
            rows_scores.append({
                "phase": phase_name,
                "algorithm": algorithm_name,
                "zone_id": int(zid) if isinstance(zid, (int, np.integer)) else zid,
                "node": members[0],
                "intra_zone_score": 0.0,
                "is_captain": 1,
            })
            continue

        zone_matrix = df_hybrid_local.loc[members, members]
        local_scores = zone_matrix.sum(axis=1)  # EXACT same score definition as your captain selection/plot
        local_scores = local_scores.sort_values(ascending=False)

        for node_str, score in local_scores.items():
            rows_scores.append({
                "phase": phase_name,
                "algorithm": algorithm_name,
                "zone_id": int(zid) if isinstance(zid, (int, np.integer)) else zid,
                "node": str(node_str),
                "intra_zone_score": float(score),
                "is_captain": int(str(node_str) == str(captain_node)) if captain_node is not None else 0,
            })

    df_scores = pd.DataFrame(rows_scores)
    df_scores.to_csv(os.path.join(out_dir, f"zone_scores_{safe_phase}_{safe_algo}.csv"), index=False)

    print(f"[CSV SAVED] {out_dir}/memberships_{safe_phase}_{safe_algo}.csv")
    print(f"[CSV SAVED] {out_dir}/zone_scores_{safe_phase}_{safe_algo}.csv")

# =================================================================
# --- ANALYSIS DRIVERS ---
# =================================================================

def run_community_detection(df_hybrid, phase_name, hybrid_matrix_filename):
    """
    Encapsulates the graph creation and community detection steps.
    Returns:
      louvain_captains_list, leiden_captains_list,
      partition_louvain_str, partition_leiden_str,
      zones_louvain, zones_leiden
    """
    total_hybrid_scores = df_hybrid.sum(axis=1)
    df_inverse_weights = df_hybrid.map(invert_correlation_weights)
    G_weighted = nx.from_pandas_adjacency(df_inverse_weights)

    # prepare mappings to/from integer node labels for centrality calculations
    node_names_str = sorted(G_weighted.nodes(), key=lambda x: int(x))
    try:
        node_names_int = [int(n) for n in node_names_str]
        mapping = dict(zip(node_names_str, node_names_int))
        inv_mapping = {v: k for k, v in mapping.items()}
        G_weighted_int = nx.relabel_nodes(G_weighted, mapping)
    except ValueError:
        G_weighted_int = G_weighted
        inv_mapping = {n: n for n in G_weighted_int.nodes()}

    # --- Run Louvain ---
    partition_louvain_int = community_louvain.best_partition(G_weighted_int, weight='weight')
    zones_louvain = {}
    for node, zone_id in partition_louvain_int.items():
        zones_louvain.setdefault(zone_id, []).append(node)

    louvain_captains_list, louvain_zone_to_captain = get_zone_captains_with_scores(
        G_weighted_int, inv_mapping, zones_louvain, "Louvain", hybrid_matrix_filename
    )
    partition_louvain_str = {inv_mapping[node_int]: zone_id for node_int, zone_id in partition_louvain_int.items()}

    # build human-readable zones for Louvain: zone_id -> [node_strs]
    zones_louvain_str = {}
    for node_str, zid in partition_louvain_str.items():
        zones_louvain_str.setdefault(zid, []).append(node_str)
    for zid in zones_louvain_str:
        zones_louvain_str[zid] = sorted(zones_louvain_str[zid], key=lambda x: int(x) if str(x).isdigit() else x)

    # --- Run Leiden (if available) ---
    leiden_captains_list, leiden_zone_to_captain = [], {}
    partition_leiden_str = {}
    zones_leiden = {}
    zones_leiden_str = {}
    try:
        coms_leiden = cdlib.algorithms.leiden(G_weighted_int, weights='weight')
        zones_list_leiden = coms_leiden.communities
        zones_leiden = {i: list(zone) for i, zone in enumerate(zones_list_leiden)}

        leiden_captains_list, leiden_zone_to_captain = get_zone_captains_with_scores(
            G_weighted_int, inv_mapping, zones_leiden, "Leiden", hybrid_matrix_filename
        )

        # build partition_leiden_str mapping (int_node -> zone_id as string mapping)
        for zone_id, nodes_int in zones_leiden.items():
            for node_int in nodes_int:
                partition_leiden_str[inv_mapping[node_int]] = zone_id

        # build human-readable zones for Leiden
        for node_str, zid in partition_leiden_str.items():
            zones_leiden_str.setdefault(zid, []).append(node_str)
        for zid in zones_leiden_str:
            zones_leiden_str[zid] = sorted(zones_leiden_str[zid], key=lambda x: int(x) if str(x).isdigit() else x)

    except Exception as e:
        print("\nLeiden detection skipped/failed for this graph. Reason:", str(e))
        zones_leiden_str = {}

    # --- Visualizations ---
    pos = nx.spring_layout(G_weighted, k=0.3, iterations=50, seed=42, weight='weight')
    draw_community_graph(G_weighted, pos, partition_louvain_str, louvain_captains_list, "Louvain", phase_name)
    draw_captain_justification_charts(zones_louvain, "Louvain", phase_name, hybrid_matrix_filename, inv_mapping,
                                      G_weighted_int)
    
    # --- Export Louvain CSVs (membership + zone scores) ---
    export_zone_csvs(
        phase_name=phase_name,
        algorithm_name="Louvain",
        partition_str=partition_louvain_str,
        zones_str=zones_louvain_str,
        zone_to_captain=louvain_zone_to_captain,
        hybrid_matrix_filename=hybrid_matrix_filename,
        out_dir="community_outputs",
    )


    if leiden_captains_list:
        draw_community_graph(G_weighted, pos, partition_leiden_str, leiden_captains_list, "Leiden", phase_name)
        draw_captain_justification_charts(zones_leiden, "Leiden", phase_name, hybrid_matrix_filename, inv_mapping,
                                          G_weighted_int)
        
        # --- Export Leiden CSVs (membership + zone scores) ---
        export_zone_csvs(
            phase_name=phase_name,
            algorithm_name="Leiden",
            partition_str=partition_leiden_str,
            zones_str=zones_leiden_str,
            zone_to_captain=leiden_zone_to_captain,
            hybrid_matrix_filename=hybrid_matrix_filename,
            out_dir="community_outputs",
        )


    return (louvain_captains_list, leiden_captains_list,
            partition_louvain_str, partition_leiden_str,
            zones_louvain, zones_leiden,
            zones_louvain_str, zones_leiden_str)


def analyze_single_phase(phase_letter):
    """Runs the entire analysis for a SINGLE phase (A, B, or C)."""
    print("\n" + "=" * 80)
    print(f"       STARTING ANALYSIS FOR PHASE {phase_letter}")
    print("=" * 80)

    nodes_file = f'nodesMVLV_phase{phase_letter}.csv'
    conn_file = f'ConnMVLV_phase{phase_letter}.csv'
    corr_file = f'MVLVCorrelation_Matrix_MI_phase{phase_letter}.csv'

    df_hybrid = load_hybrid_df(conn_file, corr_file, nodes_file)
    if df_hybrid is None:
        return ([], [], {}, {}, {}, {}, {}, {})

    output_filename = f'hybrid_matrix_Phase{phase_letter}.csv'
    df_hybrid.to_csv(output_filename)

    results = run_community_detection(df_hybrid, f'Phase {phase_letter}', output_filename)
    (louvain_captains, leiden_captains,
     partition_louvain_str, partition_leiden_str,
     zones_louvain, zones_leiden,
     zones_louvain_str, zones_leiden_str) = results

    print(f"--- Finished Analysis for Phase {phase_letter} ---\n")

    # =======================================================
    # <<< MODIFICATION START: Print Single List of ALL Captains for this Phase >>>
    # =======================================================
    print("=" * 60)
    print(f"⭐ PHASE {phase_letter} - COMPLETE CAPTAINS LISTS (Used for Exclusivity Check) ⭐")
    print(f"  Louvain Captains ({len(louvain_captains)}): {louvain_captains}")
    if leiden_captains:
        print(f"  Leiden Captains ({len(leiden_captains)}): {leiden_captains}")
    else:
        print("  Leiden Captains: Not available.")
    print("=" * 60)
    # =======================================================
    # <<< MODIFICATION END >>>

    # Print clear summary for this phase (human-readable zone -> nodes)
    print(f"\nPhase {phase_letter} - Louvain Zones (zone_id -> [node strings]):")
    for zid, members_str in sorted(zones_louvain_str.items()):
        captain = None
        # find captain for this zone (zone_id corresponds to index in louvain_captains if consistent)
        # safer: look up which captain has that zone in partition_louvain_str
        for c in louvain_captains:
            # find node c's zone
            c_zone = partition_louvain_str.get(str(c), partition_louvain_str.get(c))
            if c_zone == zid:
                captain = c
                break
        print(f"  Zone {zid}: {members_str}  => Captain: {captain}")

    if zones_leiden_str:
        print(f"\nPhase {phase_letter} - Leiden Zones (zone_id -> [node strings]):")
        for zid, members_str in sorted(zones_leiden_str.items()):
            captain = None
            for c in leiden_captains:
                c_zone = partition_leiden_str.get(str(c), partition_leiden_str.get(c))
                if c_zone == zid:
                    captain = c
                    break
            print(f"  Zone {zid}: {members_str}  => Captain: {captain}")
    else:
        print(f"\nPhase {phase_letter} - Leiden: not available or failed for this phase.")

    return (louvain_captains, leiden_captains,
            partition_louvain_str, partition_leiden_str,
            zones_louvain, zones_leiden,
            zones_louvain_str, zones_leiden_str)
    # --- 6. Return the Captains ---
    print(f"--- Finished Analysis for Phase {phase_letter} ---")
    print(f"Louvain Placements ({num_zones_louvain} zones): {louvain_captains}")  # <--- This prints Louvain Captains
    if leiden_captains:
        print(f"Leiden Placements ({num_zones_leiden} zones):  {leiden_captains}")  # <--- This prints Leiden Captains
def analyze_combined_graph():
    """Builds the averaged combined graph (the baseline) and finds captains for both methods."""
    print("\n" + "=" * 80)
    print("       STARTING ANALYSIS FOR COMBINED (AVERAGED) GRAPH")
    print("=" * 80)

    df_hybrid_A = load_hybrid_df('ConnMVLV_phaseA.csv', 'MVLVCorrelation_Matrix3_phaseA.csv', 'nodesMVLV_phaseA.csv')
    df_hybrid_B = load_hybrid_df('ConnMVLV_phaseB.csv', 'MVLVCorrelation_Matrix3_phaseB.csv', 'nodesMVLV_phaseB.csv')
    df_hybrid_C = load_hybrid_df('ConnMVLV_phaseC.csv', 'MVLVCorrelation_Matrix3_phaseC.csv', 'nodesMVLV_phaseC.csv')

    if any(df is None for df in [df_hybrid_A, df_hybrid_B, df_hybrid_C]):
        print("Error loading individual phase data for combined analysis.")
        return ([], [], {}, {}, {}, {}, {}, {})

    df_combined_hybrid = combine_matrices(df_hybrid_A, df_hybrid_B, df_hybrid_C)
    output_filename = 'hybrid_matrix_Combined.csv'
    df_combined_hybrid.to_csv(output_filename)

    results = run_community_detection(df_combined_hybrid, 'Combined (Average)', output_filename)
    (louvain_captains, leiden_captains,
     partition_louvain_str, partition_leiden_str,
     zones_louvain, zones_leiden,
     zones_louvain_str, zones_leiden_str) = results

    # Print combined summary (human-readable)
    print("\nCombined Graph - Louvain Zones (zone_id -> [node strings]):")
    for zid, members_str in sorted(zones_louvain_str.items()):
        captain = None
        for c in louvain_captains:
            c_zone = partition_louvain_str.get(str(c), partition_louvain_str.get(c))
            if c_zone == zid:
                captain = c
                break
        print(f"  Zone {zid}: {members_str}  => Captain: {captain}")

    if zones_leiden_str:
        print("\nCombined Graph - Leiden Zones (zone_id -> [node strings]):")
        for zid, members_str in sorted(zones_leiden_str.items()):
            captain = None
            for c in leiden_captains:
                c_zone = partition_leiden_str.get(str(c), partition_leiden_str.get(c))
                if c_zone == zid:
                    captain = c
                    break
            print(f"  Zone {zid}: {members_str}  => Captain: {captain}")
    else:
        print("\nCombined Graph - Leiden: not available or failed for combined graph.")

    print(f"--- Finished Combined Graph Analysis ---")
    return (louvain_captains, leiden_captains,
            partition_louvain_str, partition_leiden_str,
            zones_louvain, zones_leiden,
            zones_louvain_str, zones_leiden_str)


# =================================================================
# --- MAIN EXECUTION BLOCK (FINAL SET LOGIC) ---
# =================================================================

# --- 1. Run the analysis for each phase (Gets A, B, C Captains & partitions) ---
(louvain_captains_A, leiden_captains_A,
 partition_louvain_A, partition_leiden_A,
 zones_louvain_A, zones_leiden_A,
 zones_louvain_A_str, zones_leiden_A_str) = analyze_single_phase('A')

(louvain_captains_B, leiden_captains_B,
 partition_louvain_B, partition_leiden_B,
 zones_louvain_B, zones_leiden_B,
 zones_louvain_B_str, zones_leiden_B_str) = analyze_single_phase('B')

(louvain_captains_C, leiden_captains_C,
 partition_louvain_C, partition_leiden_C,
 zones_louvain_C, zones_leiden_C,
 zones_louvain_C_str, zones_leiden_C_str) = analyze_single_phase('C')

# --- 2. Run the Combined Graph Analysis (Gets Combined Captains/Baseline for both methods) ---
(combined_captains_Louvain, combined_captains_Leiden,
 partition_louvain_combined, partition_leiden_combined,
 zones_louvain_combined, zones_leiden_combined,
 zones_louvain_combined_str, zones_leiden_combined_str) = analyze_combined_graph()

# --- 3. Load raw node lists for presence checks (phase membership) ---
nodes_A = set(pd.read_csv('nodesMVLV_phaseA.csv', header=None).iloc[0].astype(str).tolist())
nodes_B = set(pd.read_csv('nodesMVLV_phaseB.csv', header=None).iloc[0].astype(str).tolist())
nodes_C = set(pd.read_csv('nodesMVLV_phaseC.csv', header=None).iloc[0].astype(str).tolist())
# --- 4. Final set logic: include combined baseline captains always, add a phase's captain
#       only if that captain node is absent from the other two phase node lists.
#       Apply this for both Louvain and Leiden.
print("\n" + "=" * 80)
print("       FINAL COMPREHENSIVE CAPTAINS SELECTION (BY METHOD, PHASE-EXCLUSIVE CHECK)")
print("=" * 80)

# Convert to sets for union operations when needed
set_combined_louvain = set(combined_captains_Louvain)
set_combined_leiden = set(combined_captains_Leiden)

# =================================================================
# <<< MODIFICATION START: Print Captain Lists for each Phase >>>
# =================================================================
print("\n--- LOUVAIN: All Phase Captains (Input for Exclusivity Check) ---")
print(f"Phase A Louvain Captains ({len(louvain_captains_A)}): {louvain_captains_A}")
print(f"Phase B Louvain Captains ({len(louvain_captains_B)}): {louvain_captains_B}")
print(f"Phase C Louvain Captains ({len(louvain_captains_C)}): {louvain_captains_C}")
print("-" * 60)

print("\n--- LEIDEN: All Phase Captains (Input for Exclusivity Check) ---")
if leiden_captains_A:
    print(f"Phase A Leiden Captains ({len(leiden_captains_A)}): {leiden_captains_A}")
    print(f"Phase B Leiden Captains ({len(leiden_captains_B)}): {leiden_captains_B}")
    print(f"Phase C Leiden Captains ({len(leiden_captains_C)}): {leiden_captains_C}")
else:
    print("Leiden results not available for all phases.")
print("-" * 60)
# =================================================================
# <<< MODIFICATION END >>>


# --- LOUVAIN exclusives (phase captain is added only if absent in other phases' node lists) ---
Exclusive_A_louvain = [n for n in louvain_captains_A if (n not in nodes_B and n not in nodes_C)]
Exclusive_B_louvain = [n for n in louvain_captains_B if (n not in nodes_A and n not in nodes_C)]
Exclusive_C_louvain = [n for n in louvain_captains_C if (n not in nodes_A and n not in nodes_B)]

Final_Master_Set_Louvain = set_combined_louvain.union(set(Exclusive_A_louvain)).union(set(Exclusive_B_louvain)).union(set(Exclusive_C_louvain))
Final_Master_List_Louvain = sorted(list(Final_Master_Set_Louvain), key=lambda x: int(x) if str(x).isdigit() else str(x))

print("\nLOUVAIN:")
print(f"  Baseline Combined Captains ({len(set_combined_louvain)}): {sorted(list(set_combined_louvain), key=lambda x: int(x) if str(x).isdigit() else str(x))}")
print(f"  Exclusive Phase-A Louvain captains to add (absent in B & C): {Exclusive_A_louvain}")
print(f"  Exclusive Phase-B Louvain captains to add (absent in A & C): {Exclusive_B_louvain}")
print(f"  Exclusive Phase-C Louvain captains to add (absent in A & B): {Exclusive_C_louvain}")
print(f"  Final Louvain Master List ({len(Final_Master_List_Louvain)}): {Final_Master_List_Louvain}")

# --- LEIDEN exclusives (same logic) ---
Exclusive_A_leiden = [n for n in leiden_captains_A if (n not in nodes_B and n not in nodes_C)]
Exclusive_B_leiden = [n for n in leiden_captains_B if (n not in nodes_A and n not in nodes_C)]
Exclusive_C_leiden = [n for n in leiden_captains_C if (n not in nodes_A and n not in nodes_B)]

Final_Master_Set_Leiden = set_combined_leiden.union(set(Exclusive_A_leiden)).union(set(Exclusive_B_leiden)).union(set(Exclusive_C_leiden))
Final_Master_List_Leiden = sorted(list(Final_Master_Set_Leiden), key=lambda x: int(x) if str(x).isdigit() else str(x))

print("\nLEIDEN:")
if set_combined_leiden:
    print(f"  Baseline Combined Captains ({len(set_combined_leiden)}): {sorted(list(set_combined_leiden), key=lambda x: int(x) if str(x).isdigit() else str(x))}")
else:
    print("  Baseline Combined Captains: Leiden not available / none found.")
print(f"  Exclusive Phase-A Leiden captains to add (absent in B & C): {Exclusive_A_leiden}")
print(f"  Exclusive Phase-B Leiden captains to add (absent in A & C): {Exclusive_B_leiden}")
print(f"  Exclusive Phase-C Leiden captains to add (absent in A & B): {Exclusive_C_leiden}")
print(f"  Final Leiden Master List ({len(Final_Master_List_Leiden)}): {Final_Master_List_Leiden}")

print("\n--- Script finished ---")