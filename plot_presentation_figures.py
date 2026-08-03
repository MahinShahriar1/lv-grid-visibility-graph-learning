import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

# ============================================================
# ACM acmart-friendly DOUBLE-column settings (for 1×2 layout)
# ============================================================
FIG_W = 7.16   # ~ACM double-column width
FIG_H = 2.25   # good height for 1×2 panels
DPI   = 300

plt.rcParams.update({
    "font.family": ["Linux Libertine"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    # NOTE: no bold anywhere
})

# ============================================================
# DATA (from your LaTeX table)
# ============================================================
MODELS  = ["Proposed", "Baseline-1", "Baseline-2", "Baseline-3"]
METRICS = ["MAE", "RMSE"]

VALS = {
    "Proposed":   {"MAE":0.000615, "RMSE":0.000931},
    "Baseline-1": {"MAE":0.000684, "RMSE":0.001037},
    "Baseline-2": {"MAE":0.003867, "RMSE":0.008564},
    "Baseline-3": {"MAE":0.005079, "RMSE":0.009267},
}

COLORS = {
    "Proposed":   "#000000",
    "Baseline-1": "#1f77b4",
    "Baseline-2": "#ff7f0e",
    "Baseline-3": "#2ca02c",
}

# ============================================================
# AESTHETIC CONTROLS (fake-boxplot look)
# ============================================================
PAD_FRAC         = 0.25
BOX_HALFH_FRAC   = 0.07
WHISK_HALFH_FRAC = 0.16
BOX_HALF_WIDTH   = 0.30

MIN_SPAN_FRAC_OF_MAG = 0.02
MIN_SPAN_ABS         = 1e-9
VALUE_TEXT_SIZE      = 8

# ============================================================
# Helpers
# ============================================================
def compute_axis_limits(y_values):
    ymin, ymax = float(np.min(y_values)), float(np.max(y_values))
    span = ymax - ymin
    if span < 1e-15:
        mag = max(abs(ymin), abs(ymax), 1e-15)
        span = max(MIN_SPAN_ABS, MIN_SPAN_FRAC_OF_MAG * mag)
        ymin = ymin - 0.5 * span
        ymax = ymax + 0.5 * span
    ymin_show = ymin - PAD_FRAC * span
    ymax_show = ymax + PAD_FRAC * span
    return ymin_show, ymax_show

def fmt_value(v):
    return f"{v:.6f}"

def add_center_value(ax, x0, y0, facecolor="white"):
    ax.text(
        x0, y0, fmt_value(y0),
        ha="center", va="center",
        fontsize=VALUE_TEXT_SIZE,
        color="black",
        bbox=dict(boxstyle="round,pad=0.12", facecolor=facecolor, edgecolor="none", alpha=1.0),
        zorder=10
    )

def draw_clean_fake_box(ax, x0, y0, span, color, is_proposed=False):
    box_hh   = BOX_HALFH_FRAC * span
    whisk_hh = WHISK_HALFH_FRAC * span

    rect = Rectangle(
        (x0 - BOX_HALF_WIDTH, y0 - box_hh),
        2 * BOX_HALF_WIDTH,
        2 * box_hh,
        facecolor=color,
        edgecolor="black",
        linewidth=1.3 if is_proposed else 1.1,  # subtle emphasis only via thickness
        alpha=1.0,
        zorder=3
    )
    ax.add_patch(rect)

    # whiskers OUTSIDE the box only
    ax.plot([x0, x0], [y0 + box_hh, y0 + whisk_hh], color="black", linewidth=1.0, zorder=2)
    ax.plot([x0, x0], [y0 - whisk_hh, y0 - box_hh], color="black", linewidth=1.0, zorder=2)

    capw = 0.16
    ax.plot([x0 - capw, x0 + capw], [y0 + whisk_hh, y0 + whisk_hh], color="black", linewidth=1.0, zorder=2)
    ax.plot([x0 - capw, x0 + capw], [y0 - whisk_hh, y0 - whisk_hh], color="black", linewidth=1.0, zorder=2)

    add_center_value(ax, x0, y0, facecolor="white")

def style_axes_acm(ax):
    ax.tick_params(direction="in", length=3, width=0.8)
    ax.tick_params(axis="y", pad=1)
    ax.tick_params(axis="x", pad=1)
    for s in ax.spines.values():
        s.set_linewidth(0.8)
    ax.grid(False)
    ax.yaxis.set_major_locator(MaxNLocator(5))

# ============================================================
# Main plotting function (HORIZONTAL layout)
# ============================================================
def plot_fakebox_mae_rmse_acm(
    out_dir="figures_acm",
    out_name="proposed_vs_baselines_mae_rmse_fakebox_horizontal"
):
    os.makedirs(out_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_W, FIG_H), dpi=DPI)

    x = np.arange(len(MODELS))
    title_map = {"MAE": "MAE (15-min)", "RMSE": "RMSE (15-min)"}

    for ax, met in zip(axes, METRICS):
        yvals = np.array([VALS[m][met] for m in MODELS], dtype=float)
        ymin_show, ymax_show = compute_axis_limits(yvals)
        span = ymax_show - ymin_show

        for i, m in enumerate(MODELS):
            draw_clean_fake_box(
                ax, x[i], yvals[i], span,
                COLORS[m],
                is_proposed=(m == "Proposed")
            )

        ax.set_title(title_map[met], pad=2)
        ax.set_ylim([ymin_show, ymax_show])
        ax.set_ylabel("Value")
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS, rotation=0, ha="center")
        style_axes_acm(ax)

    # Spacing tuned for double-column width (clean, not congested)
    fig.subplots_adjust(left=0.07, right=0.995, bottom=0.22, top=0.92, wspace=0.25)

    base = os.path.join(out_dir, out_name)
    fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(base + ".png", dpi=DPI, bbox_inches="tight", pad_inches=0.01)

    plt.show()
    plt.close(fig)

    print(f"Saved: {base}.pdf and {base}.png")

if __name__ == "__main__":
    plot_fakebox_mae_rmse_acm(
        out_dir="figures_acm",
        out_name="proposed_vs_baselines_mae_rmse_fakebox_horizontal"
    )
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# ============================================================
# ACM-style settings (NO bold)
# ============================================================
DPI   = 300
FIG_W = 7.16   # double-column (recommended for grouped bars)
FIG_H = 2.25

plt.rcParams.update({
    "font.family": ["Linux Libertine"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ============================================================
# DATA (from your table)
# ============================================================
proposed = {"MAE": 0.000615, "RMSE": 0.000931}
baselines = {
    "Baseline-1": {"MAE": 0.000684, "RMSE": 0.001037},
    "Baseline-2": {"MAE": 0.003867, "RMSE": 0.008564},
    "Baseline-3": {"MAE": 0.005079, "RMSE": 0.009267},
}

labels = list(baselines.keys())

# improvement% = (1 - proposed/baseline)*100
mae_impr  = np.array([(1 - proposed["MAE"]  / baselines[k]["MAE"])  * 100 for k in labels], dtype=float)
rmse_impr = np.array([(1 - proposed["RMSE"] / baselines[k]["RMSE"]) * 100 for k in labels], dtype=float)

# ============================================================
# Plot (grouped bars)
# ============================================================
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)

x = np.arange(len(labels))
w = 0.36

bars_mae  = ax.bar(x - w/2, mae_impr,  width=w, label="MAE")
bars_rmse = ax.bar(x + w/2, rmse_impr, width=w, label="RMSE")

# labels (NO title)
ax.set_ylabel("Improvement (%)")
ax.set_xlabel("Baseline")

# spines/ticks like your ACM plots, but remove "lines" you disliked
ax.tick_params(axis="y", direction="in",  length=3, width=0.8, pad=1)
ax.tick_params(axis="x", direction="out", length=0, width=0.8, pad=1)  # no x tick marks

for s in ax.spines.values():
    s.set_linewidth(0.8)

ax.spines["top"].set_visible(False)    # remove top line
ax.spines["right"].set_visible(False)  # remove right line

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=0, ha="center")

ax.yaxis.set_major_locator(MaxNLocator(5))
ax.grid(False)

# y-limit so labels fit
ymax = float(max(mae_impr.max(), rmse_impr.max()))
ax.set_ylim(0, ymax * 1.12)

# legend (clean, no box)
ax.legend(loc="upper left", frameon=False, handlelength=1.6)

# put value labels WITH % sign on top of each bar
def annotate(bars, vals):
    for rect, v in zip(bars, vals):
        ax.text(
            rect.get_x() + rect.get_width()/2,
            rect.get_height() + 0.02*ymax,
            f"{v:.2f}%",
            ha="center", va="bottom",
            fontsize=8
        )

annotate(bars_mae, mae_impr)
annotate(bars_rmse, rmse_impr)

fig.subplots_adjust(left=0.07, right=0.995, bottom=0.28, top=0.95)

# ============================================================
# Save (optional) — comment out if you don't want saving
# ============================================================
out_dir = "figures_acm"
out_name = "mae_rmse_improvement_bar"
os.makedirs(out_dir, exist_ok=True)
base = os.path.join(out_dir, out_name)
fig.savefig(base + ".pdf", bbox_inches="tight", pad_inches=0.01)
fig.savefig(base + ".png", dpi=DPI, bbox_inches="tight", pad_inches=0.01)

plt.show()
plt.close(fig)
