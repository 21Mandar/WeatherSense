"""
Phase 2 Stage 6 (2/3): generate 4 visualizations from the master CSV.

Outputs:
- docs/images/phase2/<name>.png   for README embedding
- docs/charts/<name>.html         for interactive viewing

The 4 charts:
1. mAP@50 heatmap (3 models x 13 conditions) - hero chart
2. Model robustness comparison (dual panel: heavy vs averaged)
3. Intensity progression curves (4 panels, one per weather type)
4. Per-class degradation heatmap for yolov8m (10 classes x 13 conditions)
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.detection.bdd_dataset import BDD_CLASSES


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MASTER_CSV = PROJECT_ROOT / "results" / "phase2_master_results.csv"

PNG_DIR = PROJECT_ROOT / "docs" / "images" / "phase2"
HTML_DIR = PROJECT_ROOT / "docs" / "charts"
PNG_DIR.mkdir(parents=True, exist_ok=True)
HTML_DIR.mkdir(parents=True, exist_ok=True)


# Condition display order (controls all chart axes)
CONDITION_ORDER = [
    "clean",
    "rain_light", "rain_medium", "rain_heavy",
    "fog_light", "fog_medium", "fog_heavy",
    "night_light", "night_medium", "night_heavy",
    "motion_blur_light", "motion_blur_medium", "motion_blur_heavy",
]
MODEL_ORDER = ["yolov8n", "yolov8s", "yolov8m"]
WEATHER_TYPES = ["rain", "fog", "night", "motion_blur"]
INTENSITY_ORDER = ["light", "medium", "heavy"]

# Color schemes - keep consistent across all charts
MODEL_COLORS = {
    "yolov8n": "#3b82f6",  # blue
    "yolov8s": "#f59e0b",  # amber
    "yolov8m": "#10b981",  # green
}
WEATHER_COLORS = {
    "rain":        "#3b82f6",  # blue
    "fog":         "#a78bfa",  # purple
    "night":       "#475569",  # slate
    "motion_blur": "#f59e0b",  # amber
}


def load_master():
    """Load master CSV as a list of dicts."""
    rows = []
    with open(MASTER_CSV) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def get_value(rows, model, condition, field="map_50"):
    """Find the value for a (model, condition) pair. Returns float or NaN."""
    for row in rows:
        if row["model"] == model and row["condition"] == condition:
            val = row[field]
            return float(val) if val else float("nan")
    return float("nan")


# ====================================================================
# Chart 1: mAP@50 heatmap (3 models x 13 conditions)
# ====================================================================

def chart1_heatmap(rows):
    """Heatmap: rows = conditions, cols = models, color = mAP@50."""
    print("Chart 1: mAP@50 heatmap ...")

    # Build the value matrix: (conditions x models)
    matrix = np.array([
        [get_value(rows, model, cond) for model in MODEL_ORDER]
        for cond in CONDITION_ORDER
    ])

    # ----- PNG (matplotlib) -----
    fig, ax = plt.subplots(figsize=(7, 9))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.30)

    # Cell annotations
    for i, cond in enumerate(CONDITION_ORDER):
        for j, model in enumerate(MODEL_ORDER):
            val = matrix[i, j]
            text_color = "black" if val > 0.10 else "white"
            ax.text(j, i, f"{val:.3f}",
                    ha="center", va="center", color=text_color, fontsize=10)

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, fontsize=11)
    ax.set_yticks(range(len(CONDITION_ORDER)))
    ax.set_yticklabels(CONDITION_ORDER, fontsize=10)
    ax.set_title("YOLOv8 detection performance under weather conditions\nmAP@50 on BDD100K validation set (10,000 images)",
                 fontsize=12, pad=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("mAP@50", fontsize=10)
    plt.tight_layout()
    plt.savefig(PNG_DIR / "01_heatmap_map50.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   PNG  -> {PNG_DIR / '01_heatmap_map50.png'}")

    # ----- HTML (plotly) -----
    fig_html = go.Figure(data=go.Heatmap(
        z=matrix,
        x=MODEL_ORDER,
        y=CONDITION_ORDER,
        colorscale="RdYlGn",
        zmin=0, zmax=0.30,
        text=[[f"{v:.3f}" for v in row] for row in matrix],
        texttemplate="%{text}",
        textfont={"size": 12},
        colorbar=dict(title="mAP@50"),
        hovertemplate="<b>%{y}</b><br>%{x}<br>mAP@50: %{z:.4f}<extra></extra>",
    ))
    fig_html.update_layout(
        title="YOLOv8 detection performance under weather conditions<br><sub>mAP@50 on BDD100K validation set (10,000 images)</sub>",
        xaxis_title="Model",
        yaxis_title="Condition",
        yaxis=dict(autorange="reversed"),
        width=600, height=750,
    )
    fig_html.write_html(HTML_DIR / "01_heatmap_map50.html")
    print(f"   HTML -> {HTML_DIR / '01_heatmap_map50.html'}")


# ====================================================================
# Chart 2: Model robustness comparison (dual panel)
# ====================================================================

def chart2_model_robustness(rows):
    """Two panels: (a) degradation at HEAVY intensity, (b) averaged across intensities.
    For each weather type, show degradation% per model.
    """
    print("Chart 2: Model robustness comparison ...")

    # Heavy: degradation_pct at heavy intensity, per (weather_type, model)
    heavy_data = {wt: [] for wt in WEATHER_TYPES}
    # Average: mean degradation_pct across {light, medium, heavy} per (weather_type, model)
    avg_data = {wt: [] for wt in WEATHER_TYPES}

    for wt in WEATHER_TYPES:
        for model in MODEL_ORDER:
            # Heavy
            heavy_val = get_value(rows, model, f"{wt}_heavy", "degradation_pct")
            heavy_data[wt].append(heavy_val)
            # Averaged across 3 intensities
            avg_vals = [
                get_value(rows, model, f"{wt}_{intensity}", "degradation_pct")
                for intensity in INTENSITY_ORDER
            ]
            avg_data[wt].append(np.mean(avg_vals))

    # ----- PNG (matplotlib) -----
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(WEATHER_TYPES))
    width = 0.25

    for j, model in enumerate(MODEL_ORDER):
        heavy_vals = [heavy_data[wt][j] for wt in WEATHER_TYPES]
        ax1.bar(x + (j - 1) * width, heavy_vals, width,
                label=model, color=MODEL_COLORS[model])

    ax1.set_xticks(x)
    ax1.set_xticklabels(WEATHER_TYPES, fontsize=11)
    ax1.set_ylabel("mAP@50 degradation (%)", fontsize=11)
    ax1.set_title("(a) Heavy intensity\nWorst-case degradation per weather type", fontsize=11)
    ax1.legend(loc="upper left", fontsize=10)
    ax1.grid(True, axis="y", alpha=0.3)
    ax1.set_ylim(0, 100)

    for j, model in enumerate(MODEL_ORDER):
        avg_vals = [avg_data[wt][j] for wt in WEATHER_TYPES]
        ax2.bar(x + (j - 1) * width, avg_vals, width,
                label=model, color=MODEL_COLORS[model])

    ax2.set_xticks(x)
    ax2.set_xticklabels(WEATHER_TYPES, fontsize=11)
    ax2.set_ylabel("mAP@50 degradation (%)", fontsize=11)
    ax2.set_title("(b) Averaged across intensities\nTypical-case degradation per weather type", fontsize=11)
    ax2.legend(loc="upper left", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_ylim(0, 100)

    fig.suptitle("Model size vs. weather robustness", fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(PNG_DIR / "02_model_robustness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   PNG  -> {PNG_DIR / '02_model_robustness.png'}")

    # ----- HTML (plotly) -----
    fig_html = make_subplots(
        rows=1, cols=2,
        subplot_titles=("(a) Heavy intensity", "(b) Averaged across intensities"),
        shared_yaxes=True,
    )
    for j, model in enumerate(MODEL_ORDER):
        heavy_vals = [heavy_data[wt][j] for wt in WEATHER_TYPES]
        fig_html.add_trace(
            go.Bar(name=model, x=WEATHER_TYPES, y=heavy_vals,
                   marker_color=MODEL_COLORS[model], legendgroup=model,
                   hovertemplate=f"<b>{model}</b><br>%{{x}}<br>Degradation: %{{y:.1f}}%<extra></extra>"),
            row=1, col=1,
        )
        avg_vals = [avg_data[wt][j] for wt in WEATHER_TYPES]
        fig_html.add_trace(
            go.Bar(name=model, x=WEATHER_TYPES, y=avg_vals,
                   marker_color=MODEL_COLORS[model], legendgroup=model, showlegend=False,
                   hovertemplate=f"<b>{model}</b><br>%{{x}}<br>Degradation: %{{y:.1f}}%<extra></extra>"),
            row=1, col=2,
        )

    fig_html.update_layout(
        title="Model size vs. weather robustness",
        barmode="group",
        height=500, width=1000,
        yaxis_title="mAP@50 degradation (%)",
        yaxis2_title="mAP@50 degradation (%)",
    )
    fig_html.write_html(HTML_DIR / "02_model_robustness.html")
    print(f"   HTML -> {HTML_DIR / '02_model_robustness.html'}")


# ====================================================================
# Chart 3: Intensity progression curves (4 panels, one per weather type)
# ====================================================================

def chart3_intensity_curves(rows):
    """4 panels: each shows mAP@50 vs intensity (clean -> light -> medium -> heavy)
    with one line per model.
    """
    print("Chart 3: Intensity progression curves ...")

    x_labels = ["clean", "light", "medium", "heavy"]

    # ----- PNG (matplotlib) -----
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    axes = axes.flatten()

    for i, wt in enumerate(WEATHER_TYPES):
        ax = axes[i]
        for model in MODEL_ORDER:
            ys = [
                get_value(rows, model, "clean"),
                get_value(rows, model, f"{wt}_light"),
                get_value(rows, model, f"{wt}_medium"),
                get_value(rows, model, f"{wt}_heavy"),
            ]
            ax.plot(x_labels, ys, marker="o", markersize=8, linewidth=2,
                    color=MODEL_COLORS[model], label=model)
        ax.set_title(f"{wt}", fontsize=12)
        ax.set_ylabel("mAP@50", fontsize=10)
        ax.set_ylim(0, 0.32)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", fontsize=9)

    fig.suptitle("Detection degradation across weather intensities",
                 fontsize=13, y=1.00)
    plt.tight_layout()
    plt.savefig(PNG_DIR / "03_intensity_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   PNG  -> {PNG_DIR / '03_intensity_curves.png'}")

    # ----- HTML (plotly) -----
    fig_html = make_subplots(
        rows=2, cols=2,
        subplot_titles=WEATHER_TYPES,
        shared_yaxes=True, shared_xaxes=True,
    )
    for i, wt in enumerate(WEATHER_TYPES):
        row = (i // 2) + 1
        col = (i % 2) + 1
        for model in MODEL_ORDER:
            ys = [
                get_value(rows, model, "clean"),
                get_value(rows, model, f"{wt}_light"),
                get_value(rows, model, f"{wt}_medium"),
                get_value(rows, model, f"{wt}_heavy"),
            ]
            fig_html.add_trace(
                go.Scatter(x=x_labels, y=ys, mode="lines+markers",
                           name=model, legendgroup=model,
                           showlegend=(i == 0),
                           line=dict(color=MODEL_COLORS[model], width=3),
                           marker=dict(size=10),
                           hovertemplate=f"<b>{model}</b><br>{wt} %{{x}}<br>mAP@50: %{{y:.4f}}<extra></extra>"),
                row=row, col=col,
            )

    fig_html.update_layout(
        title="Detection degradation across weather intensities",
        height=700, width=1000,
    )
    fig_html.update_yaxes(range=[0, 0.32], title_text="mAP@50")
    fig_html.write_html(HTML_DIR / "03_intensity_curves.html")
    print(f"   HTML -> {HTML_DIR / '03_intensity_curves.html'}")


# ====================================================================
# Chart 4: Per-class degradation heatmap (yolov8m, 10 classes x 13 conditions)
# ====================================================================

def chart4_per_class_heatmap(rows):
    """Heatmap of per-class AP@50:95 for yolov8m across all 13 conditions."""
    print("Chart 4: Per-class heatmap (yolov8m) ...")

    # Build matrix: (classes x conditions)
    matrix = np.zeros((len(BDD_CLASSES), len(CONDITION_ORDER)))
    for j, cond in enumerate(CONDITION_ORDER):
        for row in rows:
            if row["model"] == "yolov8m" and row["condition"] == cond:
                for i, cls in enumerate(BDD_CLASSES):
                    val = row[cls]
                    matrix[i, j] = float(val) if val else float("nan")
                break

    # ----- PNG (matplotlib) -----
    fig, ax = plt.subplots(figsize=(14, 6))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=0.40)

    for i in range(len(BDD_CLASSES)):
        for j in range(len(CONDITION_ORDER)):
            val = matrix[i, j]
            if np.isnan(val):
                ax.text(j, i, "—", ha="center", va="center",
                        color="gray", fontsize=8)
            else:
                text_color = "white" if val < 0.05 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        color=text_color, fontsize=8)

    ax.set_xticks(range(len(CONDITION_ORDER)))
    ax.set_xticklabels(CONDITION_ORDER, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(BDD_CLASSES)))
    ax.set_yticklabels(BDD_CLASSES, fontsize=10)
    ax.set_title("Per-class detection breakdown - YOLOv8m on BDD100K val\n"
                 "AP@50:95 by object class across weather conditions",
                 fontsize=12, pad=12)
    cbar = plt.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("AP@50:95", fontsize=10)
    plt.tight_layout()
    plt.savefig(PNG_DIR / "04_per_class_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"   PNG  -> {PNG_DIR / '04_per_class_heatmap.png'}")

    # ----- HTML (plotly) -----
    text_matrix = [
        ["—" if np.isnan(v) else f"{v:.2f}" for v in row_vals]
        for row_vals in matrix
    ]
    fig_html = go.Figure(data=go.Heatmap(
        z=matrix,
        x=CONDITION_ORDER,
        y=BDD_CLASSES,
        colorscale="RdYlGn",
        zmin=0, zmax=0.40,
        text=text_matrix,
        texttemplate="%{text}",
        textfont={"size": 11},
        colorbar=dict(title="AP@50:95"),
        hovertemplate="<b>%{y}</b><br>%{x}<br>AP@50:95: %{z:.4f}<extra></extra>",
    ))
    fig_html.update_layout(
        title="Per-class detection breakdown — YOLOv8m on BDD100K val<br><sub>AP@50:95 by object class across weather conditions</sub>",
        xaxis_title="Condition",
        yaxis_title="Object class",
        width=1100, height=500,
    )
    fig_html.write_html(HTML_DIR / "04_per_class_heatmap.html")
    print(f"   HTML -> {HTML_DIR / '04_per_class_heatmap.html'}")


def main():
    print(f"Loading {MASTER_CSV} ...")
    rows = load_master()
    print(f"  {len(rows)} rows loaded\n")

    chart1_heatmap(rows)
    print()
    chart2_model_robustness(rows)
    print()
    chart3_intensity_curves(rows)
    print()
    chart4_per_class_heatmap(rows)

    print(f"\n✓ All 4 charts generated.")
    print(f"  PNGs  in: {PNG_DIR}")
    print(f"  HTMLs in: {HTML_DIR}")


if __name__ == "__main__":
    main()