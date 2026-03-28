
import matplotlib.pyplot as plt

PRAB_COLORS = {
    "target": "#1f77b4",
    "nn": "#e67e22",
    "gs": "#2ca02c",
}


def setup_prab_style(
    base_fontsize=13,
    title_fontsize=14,
    legend_fontsize=11,
    figure_dpi=120,
    savefig_dpi=300,
):
    """
    General PRAB/PRR-like plotting style.

    - serif / Times-like typography
    - clean axes, inward ticks
    - moderate line widths and light dashed grid
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [
            "Times New Roman",
            "Times",
            "Nimbus Roman No9 L",
            "DejaVu Serif",
        ],
        "mathtext.fontset": "stix",
        "font.size": base_fontsize,
        "axes.titlesize": title_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": legend_fontsize,
        "figure.titlesize": title_fontsize + 1,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 5,
        "ytick.major.size": 5,
        "xtick.minor.size": 3,
        "ytick.minor.size": 3,
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "legend.frameon": False,
        "grid.alpha": 0.18,
        "grid.linestyle": "--",
        "grid.linewidth": 0.8,
        "savefig.dpi": savefig_dpi,
        "figure.dpi": figure_dpi,
        "axes.unicode_minus": False,
    })
