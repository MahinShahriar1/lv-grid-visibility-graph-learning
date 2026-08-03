import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import cdlib.algorithms  # type: ignore
import warnings

# Suppress the FutureWarning from pandas/numpy operations
warnings.simplefilter(action='ignore', category=FutureWarning)

# ==========================
# CONSENSUS SETTINGS
# ==========================
N_CONSENSUS_RUNS = 100        # you can set 70–100
CONSENSUS_THRESHOLD = 0.2     # keep edges with C_ij >= 0.2

# ==========================
# CAPTAIN (BUDGET) SETTINGS
# ==========================
BUDGET_CAPTAINS = 50        # <-- SET YOUR BUDGET HERE
USE_ABS_CORR_FOR_SCORE = True # use |corr| for global correlation score


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
    Loads one phase physical adjacency (Conn) and correlation matrix (Corr).

    IMPORTANT (as you requested):
      - Partitioning uses ONLY physical adjacency (0/1): df_hybrid = df_conn
      - Correlation is used later ONLY for captain scoring.
    """
    try:
        node_names = pd.read_csv(nodes_file, header=None).iloc[0].astype(str).tolist()

        df_conn = pd.read_csv(conn_file, header=None)
        df_conn.index = node_names
        df_conn.columns = node_names

        # We still load corr so you can validate/keep files consistent,
        # but we DO NOT use it for graph partitioning.
        df_corr = pd.read_csv(corr_file, header=None)
        df_corr.index = node_names
        df_corr.columns = node_names

        df_hybrid = df_conn * np.abs(df_corr)  # physical-only for Leiden graph
        # df_hybrid = df_conn.astype(float)
        np.fill_diagonal(df_hybrid.values, 0)
        return df_hybrid
    except Exception as e:
        print(f"Error processing files {conn_file}, {corr_file}: {e}")
        return None


def combine_matrices_sum(df_a, df_b, df_c):
    """SUM combine (kept as-is from your file)."""
    all_nodes = load_all_unique_nodes()

    A = df_a.reindex(index=all_nodes, columns=all_nodes).fillna(0)
    B = df_b.reindex(index=all_nodes, columns=all_nodes).fillna(0)
    C = df_c.reindex(index=all_nodes, columns=all_nodes).fillna(0)

    df_combined = A.add(B).add(C)

    np.fill_diagonal(df_combined.values, 0)
    return df_combined


def save_labeled_corr_matrix(corr_file, nodes_file, out_csv):
    """
    Reads raw correlation CSV (no headers), labels it with node list, and saves as labeled CSV.
    This allows captain scoring to read a single clean labeled matrix.
    """
    node_names = pd.read_csv(nodes_file, header=None).iloc[0].astype(str).tolist()
    df_corr = pd.read_csv(corr_file, header=None)
    df_corr.index = node_names
    df_corr.columns = node_names
    np.fill_diagonal(df_corr.values, 0)
    df_corr.to_csv(out_csv)
    return out_csv


def build_and_save_combined_corr(out_csv='corr_matrix_Combined.csv'):
    """
    Builds a 459x459 combined correlation matrix for GLOBAL scoring:
      Corr_combined(i,j) = max( |CorrA|, |CorrB|, |CorrC| ) after aligning to all nodes.
    Saves as labeled CSV and returns filename.
    """
    all_nodes = load_all_unique_nodes()
    if all_nodes is None:
        return None

    nodesA = pd.read_csv('nodesMVLV_phaseA.csv', header=None).iloc[0].astype(str).tolist()
    nodesB = pd.read_csv('nodesMVLV_phaseB.csv', header=None).iloc[0].astype(str).tolist()
    nodesC = pd.read_csv('nodesMVLV_phaseC.csv', header=None).iloc[0].astype(str).tolist()

    corrA = pd.read_csv('MVLVCorrelation_Matrix_MI_phaseA.csv', header=None)
    corrA.index = nodesA; corrA.columns = nodesA
    corrB = pd.read_csv('MVLVCorrelation_Matrix_MI_phaseB.csv', header=None)
    corrB.index = nodesB; corrB.columns = nodesB
    corrC = pd.read_csv('MVLVCorrelation_Matrix_MI_phaseC.csv', header=None)
    corrC.index = nodesC; corrC.columns = nodesC

    corrA = corrA.reindex(index=all_nodes, columns=all_nodes).fillna(0)
    corrB = corrB.reindex(index=all_nodes, columns=all_nodes).fillna(0)
    corrC = corrC.reindex(index=all_nodes, columns=all_nodes).fillna(0)
    
    if USE_ABS_CORR_FOR_SCORE:
        corrA = corrA.abs()
        corrB = corrB.abs()
        corrC = corrC.abs()

    A = corrA.values
    B = corrB.values
    C = corrC.values

    sum_vals = A + B + C

    # Count how many phases actually have a value for each pair.
    # Using "nonzero means present" (works if missing truly becomes 0).
    counts = (A != 0).astype(int) + (B != 0).astype(int) + (C != 0).astype(int)

    # Avoid divide-by-zero: where count==0, keep 0
    avg_vals = np.divide(sum_vals, counts, out=np.zeros_like(sum_vals), where=(counts != 0))

    corr_combined = pd.DataFrame(avg_vals, index=all_nodes, columns=all_nodes)
    np.fill_diagonal(corr_combined.values, 0)
    corr_combined.to_csv(out_csv)
    return out_csv



# =================================================================
# --- NEW CAPTAIN SELECTION (GLOBAL CORRELATION + BUDGET) ---
# =================================================================

def get_zone_captains_with_scores(G_cost_int, inv_mapping, zones, algorithm_name,
                                  corr_matrix_filename, budget=BUDGET_CAPTAINS):
    """
    Captain definition (as you requested):
      - Score(node) = GLOBAL correlation strength = sum_j Corr[node, j] over ALL nodes (j != node)
        (optionally using |corr| via USE_ABS_CORR_FOR_SCORE)
      - Baseline: 1 captain per zone = highest global score inside that zone
      - Budget:
          * if budget < #zones: pick top 'budget' among baseline captains
          * if budget > #zones: add 2nd-best per zone, then 3rd-best, ... until filled
        Tie-breaker everywhere: larger zone size.

    Returns:
      captain_list: selected nodes (strings), sorted by (score desc, zone_size desc)
      zone_to_captain: baseline captain per zone (rank-1 only)
    """
    print("\n" + "=" * 60)
    print(f" Finding 'Captains' for {algorithm_name} using GLOBAL correlation score")
    print(f"   budget={budget}, abs_corr={USE_ABS_CORR_FOR_SCORE}")
    print("=" * 60)

    # Load labeled correlation matrix
    df_corr = pd.read_csv(corr_matrix_filename, index_col=0)
    df_corr.index = df_corr.index.map(str)
    df_corr.columns = df_corr.columns.map(str)

    if USE_ABS_CORR_FOR_SCORE:
        df_corr = df_corr.abs()

    # Global correlation strength
    global_scores = df_corr.sum(axis=1).to_dict()

    # Zone nodes (strings) + zone sizes
    zones_str = {}
    zone_sizes = {}
    for zid, node_list_int in zones.items():
        node_list_str = [inv_mapping[n] for n in node_list_int]
        node_list_str = sorted(list(dict.fromkeys(node_list_str)), key=lambda x: int(x))
        zones_str[zid] = node_list_str
        zone_sizes[zid] = len(node_list_str)

    Z = len(zones_str)
    if Z == 0:
        return [], {}

    # Rank nodes per zone by global corr score
    zone_ranked = {}
    for zid, nodes in zones_str.items():
        zone_ranked[zid] = sorted(
            nodes,
            key=lambda n: (global_scores.get(n, 0.0), zone_sizes[zid]),
            reverse=True
        )

    # Round 1: best per zone
    round1 = []
    for zid in zones_str:
        if zone_ranked[zid]:
            n = zone_ranked[zid][0]
            round1.append((global_scores.get(n, 0.0), zone_sizes[zid], zid, n, 1))  # (score, zsize, zone, node, rank)

    chosen = []
    chosen_nodes = set()

    if budget <= Z:
        # pick top budget among baseline captains
        round1.sort(key=lambda x: (x[0], x[1]), reverse=True)
        chosen = round1[:budget]
        chosen_nodes = set([x[3] for x in chosen])
    else:
        # keep all baseline, then fill extras by 2nd/3rd/...
        chosen = list(round1)
        chosen_nodes = set([x[3] for x in chosen])
        remaining = budget - len(chosen)

        r = 2
        while remaining > 0:
            pool = []
            for zid in zones_str:
                if len(zone_ranked[zid]) >= r:
                    n = zone_ranked[zid][r - 1]
                    if n not in chosen_nodes:
                        pool.append((global_scores.get(n, 0.0), zone_sizes[zid], zid, n, r))
            if not pool:
                break

            pool.sort(key=lambda x: (x[0], x[1]), reverse=True)

            for item in pool:
                if remaining <= 0:
                    break
                if item[3] in chosen_nodes:
                    continue
                chosen.append(item)
                chosen_nodes.add(item[3])
                remaining -= 1

            r += 1

    chosen.sort(key=lambda x: (x[0], x[1]), reverse=True)

    captain_list = [x[3] for x in chosen]

    # baseline captain per zone
    zone_to_captain = {}
    for sc, zsz, zid, node, rank in chosen:
        if rank == 1 and zid not in zone_to_captain:
            zone_to_captain[zid] = node

    print(f"\nZones={Z}, selected captains={len(captain_list)}")
    if len(chosen) > 0:
        print("Top selected (score, zone_size, zone, node, rank_in_zone):")
        for item in chosen[:min(15, len(chosen))]:
            print("  ", item)
        if len(chosen) > 15:
            print("  ... (top 15 shown)")

    return captain_list, zone_to_captain


# =================================================================
# --- VISUALIZATION FUNCTIONS ---
# =================================================================

def draw_community_graph(G_sim, pos, partition, captains, algorithm_name, phase_name):
    """Draws the community graph (SIMILARITY graph) with captains highlighted."""
    print(f"Drawing {algorithm_name} network graph for {phase_name}...")
    plt.figure(figsize=(12, 12))
    num_zones = len(set(partition.values())) if partition else 1
    cmap = plt.cm.get_cmap('tab20', num_zones)
    colors = [cmap(partition.get(node, 0)) for node in G_sim.nodes()]
    sizes  = [1500 if node in set(captains) else 400 for node in G_sim.nodes()]

    weights = np.array([G_sim.edges[u, v].get('weight', 0.0) for u, v in G_sim.edges()], dtype=float)
    if weights.size > 0:
        w_scaled = 0.5 * (weights / (weights.max() + 1e-9))
    else:
        w_scaled = 0.2

    nx.draw_networkx_edges(G_sim, pos, alpha=0.25, width=w_scaled, edge_color='gray')
    nx.draw_networkx_nodes(G_sim, pos, node_color=colors, node_size=sizes, edgecolors='black', linewidths=0.5)
    nx.draw_networkx_labels(G_sim, pos, font_size=9)
    plt.title(f'Network Communities ({algorithm_name}) - {phase_name}', fontsize=16)
    plt.axis('off')
    plt.tight_layout()
    safe_name = phase_name.replace(" ", "_")
    plt.savefig(f'community_plot_{safe_name}_{algorithm_name.lower()}.png', dpi=300)
    plt.close()


def draw_captain_justification_charts(zones, algorithm_name, phase_name, corr_matrix_filename,
                                      inv_mapping, selected_captains):
    """
    Justification charts updated to match NEW captain rule:
      - Bars show GLOBAL correlation score (within a zone’s nodes)
      - Captain highlighted in red (rank-1 captain for that zone, if present in selected_captains)
    """
    print(f"Drawing {algorithm_name} captain justification charts for {phase_name}...")
    if not zones:
        print("  ...no zones to draw.")
        return

    df_corr = pd.read_csv(corr_matrix_filename, index_col=0)
    df_corr.index = df_corr.index.map(str)
    df_corr.columns = df_corr.columns.map(str)
    if USE_ABS_CORR_FOR_SCORE:
        df_corr = df_corr.abs()

    global_scores = df_corr.sum(axis=1)

    num_zones = len(zones)
    cols = int(np.ceil(np.sqrt(num_zones)))
    rows = int(np.ceil(num_zones / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3.5))
    axes = axes.flatten()

    selected_set = set(map(str, selected_captains))

    for i, (zone_id, node_list_int) in enumerate(zones.items()):
        ax = axes[i]
        node_list_str = sorted([inv_mapping[n] for n in node_list_int], key=lambda x: int(x))

        if len(node_list_str) == 0:
            ax.axis('off')
            continue

        zone_scores = global_scores.loc[node_list_str].sort_values(ascending=False)
        zone_scores.plot(kind='bar', ax=ax, color='gray')

        captain_node = zone_scores.index[0]
        if captain_node in selected_set:
            idx = zone_scores.index.tolist().index(captain_node)
            ax.patches[idx].set_facecolor('red')

        ax.set_title(f"Zone {zone_id} (Top: {captain_node})", fontsize=9)
        ax.set_ylabel("Global Corr Score", fontsize=8)
        ax.tick_params(axis='x', rotation=90, labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    fig.suptitle(f'Captain Justification ({algorithm_name}) - {phase_name}', fontsize=14)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    safe_name = phase_name.replace(" ", "_")
    plt.savefig(f'captain_justification_{safe_name}_{algorithm_name.lower()}.png')
    plt.close()


# =================================================================
# --- CONSENSUS LEIDEN HELPERS ---
# =================================================================

def _leiden_partition_dict(G_sim_int):
    """One Leiden run -> dict node_int -> community_id."""
    coms = cdlib.algorithms.leiden(G_sim_int, weights='weight')
    part = {}
    for cid, comm in enumerate(coms.communities):
        for n in comm:
            part[n] = cid
    return part


def _consensus_matrix(nodes_int, partitions):
    n = len(nodes_int)
    idx = {node: i for i, node in enumerate(nodes_int)}
    C = np.zeros((n, n), dtype=float)

    for part in partitions:
        groups = {}
        for node, cid in part.items():
            groups.setdefault(cid, []).append(node)

        for _, gnodes in groups.items():
            inds = [idx[x] for x in gnodes if x in idx]
            if inds:
                C[np.ix_(inds, inds)] += 1.0

    C /= max(len(partitions), 1)
    np.fill_diagonal(C, 1.0)
    return C


def _stable_leiden_zones_via_consensus(G_sim_int, runs=N_CONSENSUS_RUNS, threshold=CONSENSUS_THRESHOLD):
    nodes = sorted(list(G_sim_int.nodes()))
    parts = []
    for _ in range(runs):
        try:
            parts.append(_leiden_partition_dict(G_sim_int))
        except Exception:
            pass

    if not parts:
        p = _leiden_partition_dict(G_sim_int)
        zones = {}
        for node, cid in p.items():
            zones.setdefault(cid, []).append(node)
        return zones

    C = _consensus_matrix(nodes, parts)

    Gc = nx.Graph()
    Gc.add_nodes_from(nodes)

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            w = C[i, j]
            if w >= threshold:
                Gc.add_edge(nodes[i], nodes[j], weight=float(w))

    if Gc.number_of_edges() == 0:
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                w = C[i, j]
                if w > 0:
                    Gc.add_edge(nodes[i], nodes[j], weight=float(w))

    p_final = _leiden_partition_dict(Gc)

    zones = {}
    for node, cid in p_final.items():
        zones.setdefault(cid, []).append(node)

    keys = sorted(zones.keys())
    zones2 = {i: zones[k] for i, k in enumerate(keys)}
    return zones2


# =================================================================
# --- ANALYSIS DRIVERS ---
# =================================================================

def run_community_detection(df_hybrid, phase_name, hybrid_matrix_filename, corr_matrix_filename):
    A = df_hybrid.copy()
    np.fill_diagonal(A.values, 0.0)

    G_sim = nx.from_pandas_adjacency(A)

    # cost graph kept as-is
    G_cost = nx.Graph()
    for u, v, d in G_sim.edges(data=True):
        s = float(d.get("weight", 0.0))
        if s > 0.0:
            G_cost.add_edge(u, v, length=1.0 / s)

    node_names_str = sorted(G_sim.nodes(), key=lambda x: int(x) if str(x).isdigit() else x)
    try:
        node_names_int = [int(n) for n in node_names_str]
        mapping    = dict(zip(node_names_str, node_names_int))
        inv_map    = {v: k for k, v in mapping.items()}
        G_sim_int  = nx.relabel_nodes(G_sim,  mapping)
        G_cost_int = nx.relabel_nodes(G_cost, mapping)
    except ValueError:
        G_sim_int, G_cost_int = G_sim, G_cost
        inv_map = {n: n for n in G_sim_int.nodes()}

    leiden_captains_list, partition_leiden_str = [], {}
    zones_leiden = {}

    try:
        zones_leiden = _stable_leiden_zones_via_consensus(
            G_sim_int, runs=N_CONSENSUS_RUNS, threshold=CONSENSUS_THRESHOLD
        )

        leiden_captains_list, _ = get_zone_captains_with_scores(
            G_cost_int, inv_map, zones_leiden, "Leiden", corr_matrix_filename, budget=BUDGET_CAPTAINS
        )

        for zone_id, nodes_int in zones_leiden.items():
            for node_int in nodes_int:
                partition_leiden_str[inv_map[node_int]] = zone_id

    except Exception as e:
        print("\nLeiden detection skipped/failed for this graph. Reason:", str(e))

    pos = nx.spring_layout(G_sim, k=0.3, iterations=50, seed=42, weight='weight')
    if leiden_captains_list:
        draw_community_graph(G_sim, pos, partition_leiden_str, leiden_captains_list, "Leiden", phase_name)
        draw_captain_justification_charts(
            zones_leiden, "Leiden", phase_name, corr_matrix_filename, inv_map, leiden_captains_list
        )

    return (leiden_captains_list, partition_leiden_str, zones_leiden)


def analyze_single_phase(phase_letter):
    print("\n" + "=" * 80)
    print(f"       STARTING ANALYSIS FOR PHASE {phase_letter}")
    print("=" * 80)

    nodes_file = f'nodesMVLV_phase{phase_letter}.csv'
    conn_file = f'ConnMVLV_phase{phase_letter}.csv'
    corr_file = f'MVLVCorrelation_Matrix_MI_phase{phase_letter}.csv'

    df_hybrid = load_hybrid_df(conn_file, corr_file, nodes_file)
    if df_hybrid is None:
        return ([], {}, {})

    output_filename = f'hybrid_matrix_Phase{phase_letter}.csv'
    df_hybrid.to_csv(output_filename)

    corr_out = f'corr_matrix_Phase{phase_letter}.csv'
    save_labeled_corr_matrix(corr_file, nodes_file, corr_out)

    (leiden_captains, partition_leiden_str, zones_leiden) = run_community_detection(
        df_hybrid, f'Phase {phase_letter}', output_filename, corr_out
    )

    print(f"--- Finished Analysis for Phase {phase_letter} ---\n")

    print("=" * 60)
    print(f"⭐ PHASE {phase_letter} - COMPLETE CAPTAINS LISTS (Used for Exclusivity Check) ⭐")
    if leiden_captains:
        print(f"  Leiden Captains ({len(leiden_captains)}): {leiden_captains}")
    else:
        print("  Leiden Captains: Not available.")
    print("=" * 60)

    if partition_leiden_str:
        print(f"\nPhase {phase_letter} - Leiden Zones (zone_id -> [node strings]):")
        for zid in sorted(set(partition_leiden_str.values())):
            members = sorted([n for n, z in partition_leiden_str.items() if z == zid], key=lambda x: int(x))
            captain = None
            for c in leiden_captains:
                c_zone = partition_leiden_str.get(str(c), partition_leiden_str.get(c))
                if c_zone == zid:
                    captain = c
                    break
            print(f"  Zone {zid}: {members}  => Captain: {captain}")
    else:
        print(f"\nPhase {phase_letter} - Leiden: not available or failed for this phase.")

    return (leiden_captains, partition_leiden_str, zones_leiden)


def analyze_combined_graph():
    print("\n" + "=" * 80)
    print("       STARTING ANALYSIS FOR COMBINED (SUM) GRAPH")
    print("=" * 80)

    df_hybrid_A = load_hybrid_df('ConnMVLV_phaseA.csv', 'MVLVCorrelation_Matrix_MI_phaseA.csv', 'nodesMVLV_phaseA.csv')
    df_hybrid_B = load_hybrid_df('ConnMVLV_phaseB.csv', 'MVLVCorrelation_Matrix_MI_phaseB.csv', 'nodesMVLV_phaseB.csv')
    df_hybrid_C = load_hybrid_df('ConnMVLV_phaseC.csv', 'MVLVCorrelation_Matrix_MI_phaseC.csv', 'nodesMVLV_phaseC.csv')

    if any(df is None for df in [df_hybrid_A, df_hybrid_B, df_hybrid_C]):
        print("Error loading individual phase data for combined analysis.")
        return ([], {}, {})

    df_combined_hybrid = combine_matrices_sum(df_hybrid_A, df_hybrid_B, df_hybrid_C)
    output_filename = 'hybrid_matrix_Combined.csv'
    df_combined_hybrid.to_csv(output_filename)

    corr_output_filename = build_and_save_combined_corr('corr_matrix_Combined.csv')
    if corr_output_filename is None:
        print("Error building combined correlation matrix.")
        return ([], {}, {})

    (leiden_captains, partition_leiden_str, zones_leiden) = run_community_detection(
        df_combined_hybrid, 'Combined (Sum)', output_filename, corr_output_filename
    )

    if partition_leiden_str:
        print("\nCombined Graph - Leiden Zones (zone_id -> [node strings]):")
        for zid in sorted(set(partition_leiden_str.values())):
            members = sorted([n for n, z in partition_leiden_str.items() if z == zid], key=lambda x: int(x))
            captain = None
            for c in leiden_captains:
                c_zone = partition_leiden_str.get(str(c), partition_leiden_str.get(c))
                if c_zone == zid:
                    captain = c
                    break
            print(f"  Zone {zid}: {members}  => Captain: {captain}")
    else:
        print("\nCombined Graph - Leiden: not available or failed for combined graph.")

    print(f"--- Finished Combined Graph Analysis ---")
    return (leiden_captains, partition_leiden_str, zones_leiden)


# =================================================================
# --- MAIN EXECUTION BLOCK (FINAL SET LOGIC) ---
# =================================================================

(leiden_captains_A, partition_leiden_A, zones_leiden_A) = analyze_single_phase('A')
(leiden_captains_B, partition_leiden_B, zones_leiden_B) = analyze_single_phase('B')
(leiden_captains_C, partition_leiden_C, zones_leiden_C) = analyze_single_phase('C')

(combined_captains_Leiden, partition_leiden_combined, zones_leiden_combined) = analyze_combined_graph()

nodes_A = set(pd.read_csv('nodesMVLV_phaseA.csv', header=None).iloc[0].astype(str).tolist())
nodes_B = set(pd.read_csv('nodesMVLV_phaseB.csv', header=None).iloc[0].astype(str).tolist())
nodes_C = set(pd.read_csv('nodesMVLV_phaseC.csv', header=None).iloc[0].astype(str).tolist())

print("\n" + "=" * 80)
print("       FINAL COMPREHENSIVE CAPTAINS SELECTION (LEIDEN, PHASE-EXCLUSIVE CHECK)")
print("=" * 80)

set_combined_leiden = set(combined_captains_Leiden)

print("\n--- LEIDEN: All Phase Captains (Input for Exclusivity Check) ---")
if leiden_captains_A:
    print(f"Phase A Leiden Captains ({len(leiden_captains_A)}): {leiden_captains_A}")
    print(f"Phase B Leiden Captains ({len(leiden_captains_B)}): {leiden_captains_B}")
    print(f"Phase C Leiden Captains ({len(leiden_captains_C)}): {leiden_captains_C}")
else:
    print("Leiden results not available for all phases.")
print("-" * 60)

Exclusive_A_leiden = [n for n in leiden_captains_A if (n not in nodes_B and n not in nodes_C)]
Exclusive_B_leiden = [n for n in leiden_captains_B if (n not in nodes_A and n not in nodes_C)]
Exclusive_C_leiden = [n for n in leiden_captains_C if (n not in nodes_A and n not in nodes_B)]

Final_Master_Set_Leiden = set_combined_leiden.union(set(Exclusive_A_leiden)).union(set(Exclusive_B_leiden)).union(set(Exclusive_C_leiden))
Final_Master_List_Leiden = sorted(list(set(combined_captains_Leiden)), key=lambda x: int(x))


print("\nLEIDEN:")
# if set_combined_leiden:
#     print(f"  Baseline Combined Captains ({len(set_combined_leiden)}): {sorted(list(set_combined_leiden), key=lambda x: int(x) if str(x).isdigit() else str(x))}")
# else:
#     print("  Baseline Combined Captains: Leiden not available / none found.")
# print(f"  Exclusive Phase-A Leiden captains to add (absent in B & C): {Exclusive_A_leiden}")
# print(f"  Exclusive Phase-B Leiden captains to add (absent in A & C): {Exclusive_B_leiden}")
# print(f"  Exclusive Phase-C Leiden captains to add (absent in A & B): {Exclusive_C_leiden}")
print(f"  Final Leiden Master List ({len(Final_Master_List_Leiden)}): {Final_Master_List_Leiden}")

print("\n--- Script finished ---")
