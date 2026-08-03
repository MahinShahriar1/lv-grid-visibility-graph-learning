# plot_actual_vs_four_models_with_15min_markers_ONEPLOT.py
# ONE plot:
#   Actual + GraphSAGE(Proposed) + GraphSAGE(Random) + GCN(LL) + GCN(Random)
# with 15-min scatter markers, ACM single-column style legend below.
#
# READABILITY UPDATES (per your request):
#  - Actual line is THINNER
#  - ALL model lines are SOLID and SAME thickness
#  - Legend includes: "Dots: 15 min interval"

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.lines import Line2D


# ----------------------------
# CONFIG
# ----------------------------
TS_DIR = os.path.join("outputs", "saved_timeseries")

# ---- GraphSAGE CSVs ----
CSV_GS_PROPOSED = os.path.join(TS_DIR, "node_445_phase_A_timeseries_full_actual.csv")
CSV_GS_BASELINE = os.path.join(TS_DIR, "node_445_phase_A_timeseries_full.csv")

GS_PROPOSED_LABEL = "Proposed(Modified_GraphSAGE+Louvain/Leiden)"
GS_BASELINE_LABEL = "Baseline-1(Modified_GraphSAGE+Random)"
ACTUAL_LABEL      = "Actual"

# ---- GCN CSVs ----
CSV_GCN_PROPOSED = os.path.join(TS_DIR, "node_445_phase_A_timeseries_full_GCN.csv")
CSV_GCN_BASELINE = os.path.join(TS_DIR, "node_445_phase_A_timeseries_full_GCN_random.csv")

GCN_PROPOSED_LABEL = "Baseline-2(GCN+Louvain/Leiden)"
GCN_BASELINE_LABEL = "Baseline-3(GCN+Random)"

# Plot slice
PLOT_START = 40
PLOT_END   = 100
d_interval = 15

# Output
SAVE_DIR = os.path.join("outputs", "comparison")
DPI = 300
OUT_STEM = "445A_actual_vs_4_models_200steps_GraphSAGE_and_GCN"

# ----------------------------
# FIGURE SETTINGS (ACM single-column)
# ----------------------------
FIG_W, FIG_H = 3.33, 1.8

plt.rcParams.update({
    "font.family": ["Linux Libertine"],
    "font.size": 9,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ----------------------------
# Helpers
# ----------------------------
def load_timeseries_any_schema(csv_path: str) -> pd.DataFrame:
    """
    Returns DataFrame with unified columns:
      time_step, actual_pu, pred_pu

    Supports:
      - GraphSAGE style: time_step, actual_pu, pred_final_pu
      - GCN style:       t_index, y_true, y_pred
    """
    df = pd.read_csv(csv_path)

    # time column
    if "time_step" in df.columns:
        tcol = "time_step"
    elif "t_index" in df.columns:
        tcol = "t_index"
    else:
        raise ValueError(f"[{csv_path}] missing time column. Found: {list(df.columns)}")

    # actual column
    if "actual_pu" in df.columns:
        acol = "actual_pu"
    elif "y_true" in df.columns:
        acol = "y_true"
    else:
        raise ValueError(f"[{csv_path}] missing actual column. Found: {list(df.columns)}")

    # pred column
    if "pred_final_pu" in df.columns:
        pcol = "pred_final_pu"
    elif "y_pred" in df.columns:
        pcol = "y_pred"
    else:
        raise ValueError(f"[{csv_path}] missing prediction column. Found: {list(df.columns)}")

    out = df[[tcol, acol, pcol]].copy()
    out.columns = ["time_step", "actual_pu", "pred_pu"]
    out = out.sort_values("time_step").reset_index(drop=True)
    return out


def plot_actual_vs_four_models_one_plot(
    csv_gs_prop: str,
    csv_gs_base: str,
    csv_gcn_prop: str,
    csv_gcn_base: str,
    label_gs_prop: str,
    label_gs_base: str,
    label_gcn_prop: str,
    label_gcn_base: str,
    label_actual: str,
    save_dir: str,
    out_stem: str,
    plot_start: int,
    plot_end: int,
    d_interval: int,
):
    # Load all four CSVs (schema auto-detect)
    df_gs_p   = load_timeseries_any_schema(csv_gs_prop)
    df_gs_b   = load_timeseries_any_schema(csv_gs_base)
    df_gcn_p  = load_timeseries_any_schema(csv_gcn_prop)
    df_gcn_b  = load_timeseries_any_schema(csv_gcn_base)

    # Merge on time_step (inner intersection of all)
    m = df_gs_p[["time_step", "actual_pu", "pred_pu"]].rename(
        columns={"actual_pu": "actual", "pred_pu": "pred_gs_prop"}
    )

    m = pd.merge(
        m,
        df_gs_b[["time_step", "pred_pu"]].rename(columns={"pred_pu": "pred_gs_base"}),
        on="time_step",
        how="inner",
    )

    m = pd.merge(
        m,
        df_gcn_p[["time_step", "pred_pu"]].rename(columns={"pred_pu": "pred_gcn_prop"}),
        on="time_step",
        how="inner",
    )

    m = pd.merge(
        m,
        df_gcn_b[["time_step", "pred_pu"]].rename(columns={"pred_pu": "pred_gcn_base"}),
        on="time_step",
        how="inner",
    )

    m = m.sort_values("time_step").reset_index(drop=True)

    total_len = len(m)
    ps = max(0, int(plot_start))
    pe = total_len if plot_end is None else min(total_len, int(plot_end))
    if pe <= ps:
        ps, pe = 0, min(total_len, 200)

    m_plot = m.iloc[ps:pe].copy()
    x_plot = m_plot["time_step"].to_numpy(dtype=int)

    actual        = m_plot["actual"].to_numpy(dtype=float)
    pred_gs_prop  = m_plot["pred_gs_prop"].to_numpy(dtype=float)
    pred_gs_base  = m_plot["pred_gs_base"].to_numpy(dtype=float)
    pred_gcn_prop = m_plot["pred_gcn_prop"].to_numpy(dtype=float)
    pred_gcn_base = m_plot["pred_gcn_base"].to_numpy(dtype=float)

    # 15-min markers (FIXED: anchor to true 15-min grid, not to plot_start)
    first_interval_x = ((x_plot.min() + int(d_interval) - 1) // int(d_interval)) * int(d_interval)
    interval_x = np.arange(first_interval_x, x_plot.max() + 1, int(d_interval), dtype=int)
    x_set = set(x_plot.tolist())
    interval_x = np.array([t for t in interval_x if t in x_set], dtype=int)

    lookup_a  = dict(zip(x_plot, actual))
    lookup_p1 = dict(zip(x_plot, pred_gs_prop))
    lookup_p2 = dict(zip(x_plot, pred_gs_base))
    lookup_p3 = dict(zip(x_plot, pred_gcn_prop))
    lookup_p4 = dict(zip(x_plot, pred_gcn_base))

    a_int  = np.array([lookup_a[t]  for t in interval_x], dtype=float)
    p1_int = np.array([lookup_p1[t] for t in interval_x], dtype=float)
    p2_int = np.array([lookup_p2[t] for t in interval_x], dtype=float)
    p3_int = np.array([lookup_p3[t] for t in interval_x], dtype=float)
    p4_int = np.array([lookup_p4[t] for t in interval_x], dtype=float)

    # ----------------------------
    # Plot (readability styling)
    # ----------------------------
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    MODEL_LW  = .5   # same thickness for all models
    ACTUAL_LW = .7   # thinner actual so it doesn't dominate

    # ALL models SOLID lines (same thickness)
    ax.plot(x_plot, pred_gs_prop,  label=label_gs_prop,  linestyle="-", linewidth=MODEL_LW)
    ax.plot(x_plot, pred_gs_base,  label=label_gs_base,  linestyle=":", linewidth=MODEL_LW)
    ax.plot(x_plot, pred_gcn_prop, label=label_gcn_prop, linestyle=":", linewidth=MODEL_LW)
    ax.plot(x_plot, pred_gcn_base, label=label_gcn_base, linestyle=":", linewidth=MODEL_LW)

    # Actual SOLID but thinner
    ax.plot(x_plot, actual, label=label_actual, linestyle="-", linewidth=ACTUAL_LW)

    # 15-min scatter markers (NO legend entries to avoid legend explosion)
    if interval_x.size > 0:
        ax.scatter(interval_x, p1_int, s=3, label="_nolegend_", zorder=2)
        ax.scatter(interval_x, p2_int, s=3, label="_nolegend_", zorder=2)
        ax.scatter(interval_x, p3_int, s=3, label="_nolegend_", zorder=2)
        ax.scatter(interval_x, p4_int, s=3, label="_nolegend_", zorder=2)
        ax.scatter(interval_x, a_int,  s=3, label="_nolegend_", zorder=2)

    # y-lims + ticks
    data_lo = float(min(
        actual.min(),
        pred_gs_prop.min(),
        pred_gs_base.min(),
        pred_gcn_prop.min(),
        pred_gcn_base.min()
    ))
    data_hi = float(max(
        actual.max(),
        pred_gs_prop.max(),
        pred_gs_base.max(),
        pred_gcn_prop.max(),
        pred_gcn_base.max()
    ))
    yrange = max(1e-9, data_hi - data_lo)
    pad = max(0.0002, 0.05 * yrange)
    ax.set_ylim(data_lo - pad, data_hi + pad)

    ax.yaxis.set_major_locator(MaxNLocator(5))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))

    # FIXED: put x ticks on the same 15-min anchors as the dots
    if interval_x.size > 0:
        ax.set_xticks(interval_x)
    else:
        ax.xaxis.set_major_locator(MaxNLocator(6))

    # x padding
    xspan = max(1, x_plot.max() - x_plot.min())
    xpad = int(round(0.03 * xspan))
    ax.set_xlim(x_plot.min() - xpad, x_plot.max() + xpad)

    ax.set_xlabel("Time step (min)")
    ax.set_ylabel("Voltage (p.u.)")

    ax.tick_params(direction="in", length=3, width=0.8)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    ax.grid(False)

    # ----------------------------
    # Legend (ADD dot explanation)
    # ----------------------------
    handles, labels = ax.get_legend_handles_labels()
    filtered = [(h, l) for h, l in zip(handles, labels) if l != "_nolegend_"]
    handles, labels = zip(*filtered)

    dot_handle = Line2D(
        [], [], linestyle="none", marker="o", markersize=5,
        label=f"Dots: {int(d_interval)} min interval"
    )

    handles = list(handles) + [dot_handle]
    labels  = list(labels)  + [dot_handle.get_label()]

    leg = fig.legend(
        handles, labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.03),
        ncol=2,  # 6 entries => 3 rows
        frameon=True,
        framealpha=1.0,
        facecolor="white",
        edgecolor="0.3",
        borderpad=0.20,
        labelspacing=0.20,
        handlelength=2.0,
        handletextpad=0.5,
        columnspacing=1.4,
    )
    leg.get_frame().set_linewidth(0.6)

    fig.subplots_adjust(left=0.14, right=0.995, top=0.98, bottom=0.34)
    ax.xaxis.labelpad = 2

    # Save
    os.makedirs(save_dir, exist_ok=True)
    base = os.path.join(save_dir, out_stem)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(base + ".png", dpi=DPI, bbox_inches="tight", pad_inches=0.01)

    plt.show()
    plt.close(fig)

    print(f"Saved: {base}.pdf and {base}.png")


# ----------------------------
# RUN: ONE combined plot
# ----------------------------
if __name__ == "__main__":
    plot_actual_vs_four_models_one_plot(
        csv_gs_prop=CSV_GS_PROPOSED,
        csv_gs_base=CSV_GS_BASELINE,
        csv_gcn_prop=CSV_GCN_PROPOSED,
        csv_gcn_base=CSV_GCN_BASELINE,

        label_gs_prop=GS_PROPOSED_LABEL,
        label_gs_base=GS_BASELINE_LABEL,
        label_gcn_prop=GCN_PROPOSED_LABEL,
        label_gcn_base=GCN_BASELINE_LABEL,
        label_actual=ACTUAL_LABEL,

        save_dir=SAVE_DIR,
        out_stem=OUT_STEM,
        plot_start=PLOT_START,
        plot_end=PLOT_END,
        d_interval=d_interval,
    )
