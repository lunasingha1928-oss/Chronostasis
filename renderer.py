"""
renderer.py — Chronostasis Visual Report Generator
====================================================
Generates 4 visualizations after each episode:
1. Flood risk map (high/moderate/low zones)
2. SAR flood extent bar chart (2022-2024)
3. Year-on-year flood comparison
4. District-level stats table
Returns base64-encoded PNG images.
"""

import base64
import io
import json
from typing import Any, Dict, List

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.gridspec as gridspec
    import numpy as np
    MATPLOTLIB_OK = True
except ImportError:
    MATPLOTLIB_OK = False


def _fig_to_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    buf.seek(0)
    return base64.b64encode(buf.read()).decode()


def render_flood_report(region: Dict[str, Any], episode_history: List[Dict],
                        task_id: str) -> Dict[str, str]:
    """
    Generate all 4 visualizations for a region.
    Returns dict of {chart_name: base64_png}.
    """
    if not MATPLOTLIB_OK:
        return {"error": "matplotlib not available"}

    BG    = "#060a0f"
    SURF  = "#0c1219"
    AC    = "#00d4ff"
    OK    = "#00ff88"
    WARN  = "#ff6b2b"
    MU    = "#4a7a9b"
    TEXT  = "#e8f4f8"
    RED   = "#ff4444"
    YEL   = "#ffcc00"

    results = {}

    # ── 1. FLOOD RISK ZONES PIE / BAR ──────────────────────
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    ax.set_facecolor(SURF)

    rz     = region["risk_zones_km2"]
    labels = ["High Risk", "Moderate Risk", "Low Risk"]
    values = [rz["high"], rz["moderate"], rz["low"]]
    colors = [RED, YEL, OK]

    bars = ax.barh(labels, values, color=colors, edgecolor=BG, linewidth=1.5, height=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 80, bar.get_y() + bar.get_height() / 2,
                f"{val:,.0f} km²", va="center", color=TEXT, fontsize=10, fontweight="bold")

    ax.set_xlabel("Area (km²)", color=MU, fontsize=10)
    ax.set_title(f"Flood Risk Zones — {region['name']}", color=TEXT,
                 fontsize=13, fontweight="bold", pad=12)
    ax.tick_params(colors=TEXT)
    ax.spines[:].set_color(SURF)
    ax.set_xlim(0, max(values) * 1.25)
    for spine in ax.spines.values():
        spine.set_edgecolor(MU)
        spine.set_alpha(0.3)
    ax.xaxis.label.set_color(MU)
    ax.tick_params(axis="x", colors=MU)
    ax.tick_params(axis="y", colors=TEXT, labelsize=11)

    total = sum(values)
    for bar, val, col in zip(bars, values, colors):
        pct = val / total * 100
        ax.text(bar.get_width() / 2, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", ha="center",
                color=BG, fontsize=9, fontweight="bold")

    plt.tight_layout()
    results["risk_zones"] = _fig_to_b64(fig)
    plt.close(fig)

    # ── 2. SAR FLOOD EXTENT 2022-2024 ──────────────────────
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=BG)
    ax.set_facecolor(SURF)

    fa    = region["flood_areas"]
    years = [2022, 2023, 2024]
    areas = [fa.get(yr, fa.get(str(yr), 0)) for yr in years]
    peak  = region["peak_year"]
    bar_colors = [RED if yr == peak else AC for yr in years]

    bars = ax.bar([str(y) for y in years], areas, color=bar_colors,
                  edgecolor=BG, linewidth=1.5, width=0.5)

    for bar, val, yr in zip(bars, areas, years):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 40,
                f"{val:,.0f}", ha="center", color=TEXT, fontsize=10, fontweight="bold")
        if yr == peak:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() / 2,
                    "PEAK", ha="center", va="center",
                    color=BG, fontsize=9, fontweight="bold")

    ax.set_ylabel("Flood Extent (km²)", color=MU, fontsize=10)
    ax.set_title(f"SAR Flood Extent — {region['name']} (2022–2024)",
                 color=TEXT, fontsize=13, fontweight="bold", pad=12)
    ax.set_ylim(0, max(areas) * 1.2)
    ax.tick_params(axis="x", colors=TEXT, labelsize=12)
    ax.tick_params(axis="y", colors=MU)
    ax.yaxis.label.set_color(MU)
    for spine in ax.spines.values():
        spine.set_edgecolor(MU)
        spine.set_alpha(0.3)
    ax.set_facecolor(SURF)

    legend = [mpatches.Patch(color=RED, label=f"Peak year ({peak})"),
              mpatches.Patch(color=AC,  label="Other years")]
    ax.legend(handles=legend, facecolor=SURF, edgecolor=MU,
              labelcolor=TEXT, fontsize=9)

    plt.tight_layout()
    results["sar_extent"] = _fig_to_b64(fig)
    plt.close(fig)

    # ── 3. DISTRICT CHRONIC INUNDATION TABLE ───────────────
    districts = region["chronic_districts"]
    fig, ax   = plt.subplots(figsize=(7, max(3.5, len(districts) * 0.7 + 1.5)),
                              facecolor=BG)
    ax.set_facecolor(BG)
    ax.axis("off")

    pop_per_district = region["chronic_pop"] // len(districts)
    area_per         = region["chronic_km2"] / len(districts)

    col_labels = ["District", "Flood Years", "Est. Area (km²)", "Est. Population"]
    rows = [[d, "2022, 2023, 2024", f"{area_per:,.0f}", f"{pop_per_district:,}"]
            for d in districts]
    rows.append(["TOTAL", "All 3 years",
                 f"{region['chronic_km2']:,.1f}",
                 f"{region['chronic_pop']:,}"])

    row_colors = [[SURF] * 4] * len(districts) + [["#1a2a38"] * 4]
    cell_colors = row_colors

    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", loc="center",
                   cellColours=cell_colors)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.6)

    for (row, col), cell in tbl.get_celld().items():
        cell.set_edgecolor(MU)
        if row == 0:
            cell.set_facecolor("#1a3a50")
            cell.set_text_props(color=AC, fontweight="bold")
        elif row == len(rows):
            cell.set_facecolor("#1a2a38")
            cell.set_text_props(color=YEL, fontweight="bold")
        else:
            cell.set_text_props(color=TEXT)

    ax.set_title(f"Chronically Inundated Districts — {region['name']}",
                 color=TEXT, fontsize=13, fontweight="bold", pad=16, y=0.98)
    plt.tight_layout()
    results["district_table"] = _fig_to_b64(fig)
    plt.close(fig)

    # ── 4. COMBINED SUMMARY DASHBOARD ──────────────────────
    fig = plt.figure(figsize=(12, 8), facecolor=BG)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Top-left: accuracy gauge
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(SURF)
    acc = region["accuracy_pct"]
    ax1.barh(["Model\nAccuracy"], [acc], color=OK, height=0.4)
    ax1.barh(["Model\nAccuracy"], [100 - acc], left=[acc], color="#1a2a38", height=0.4)
    ax1.text(acc / 2, 0, f"{acc:.2f}%", ha="center", va="center",
             color=BG, fontsize=14, fontweight="bold")
    ax1.set_xlim(0, 100)
    ax1.set_title("SAR Model Accuracy", color=TEXT, fontsize=11, fontweight="bold")
    ax1.tick_params(colors=MU)
    ax1.set_xlabel("%", color=MU, fontsize=9)
    for spine in ax1.spines.values():
        spine.set_edgecolor(MU)
        spine.set_alpha(0.3)

    # Top-right: flood extent bars
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(SURF)
    bar_c2 = [RED if yr == peak else AC for yr in years]
    ax2.bar([str(y) for y in years], areas, color=bar_c2, edgecolor=BG, width=0.5)
    ax2.set_title("SAR Flood Extent (km²)", color=TEXT, fontsize=11, fontweight="bold")
    ax2.tick_params(axis="x", colors=TEXT)
    ax2.tick_params(axis="y", colors=MU)
    for spine in ax2.spines.values():
        spine.set_edgecolor(MU)
        spine.set_alpha(0.3)

    # Bottom-left: risk zones
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(SURF)
    ax3.barh(["Low", "Moderate", "High"], [rz["low"], rz["moderate"], rz["high"]],
             color=[OK, YEL, RED], edgecolor=BG, height=0.5)
    ax3.set_title("Risk Zone Areas (km²)", color=TEXT, fontsize=11, fontweight="bold")
    ax3.tick_params(axis="y", colors=TEXT)
    ax3.tick_params(axis="x", colors=MU)
    for spine in ax3.spines.values():
        spine.set_edgecolor(MU)
        spine.set_alpha(0.3)

    # Bottom-right: key stats text
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(SURF)
    ax4.axis("off")
    stats = [
        ("Region",       region["name"]),
        ("River",        region["river"]),
        ("Peak year",    str(peak)),
        ("Chronic area", f"{region['chronic_km2']:,.1f} km²"),
        ("Population",   f"{region['chronic_pop']:,}"),
        ("Districts",    str(len(districts))),
        ("Rainfall",     f"{region['peak_rainfall_mm']} mm peak"),
    ]
    for i, (label, val) in enumerate(stats):
        y = 0.9 - i * 0.13
        ax4.text(0.02, y, label + ":", color=MU,
                 fontsize=10, transform=ax4.transAxes)
        ax4.text(0.45, y, val, color=TEXT, fontsize=10,
                 fontweight="bold", transform=ax4.transAxes)
    ax4.set_title("Key Statistics", color=TEXT, fontsize=11, fontweight="bold")

    fig.suptitle(f"Chronostasis Flood Intelligence Report — {region['name']}",
                 color=AC, fontsize=14, fontweight="bold", y=1.01)

    results["dashboard"] = _fig_to_b64(fig)
    plt.close(fig)

    return results
