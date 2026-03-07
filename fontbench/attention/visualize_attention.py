"""Generate publication-quality 2×3 attention heatmap figure.

Reads .npz attention maps (from extract_attention.py) and overlays them
on the original benchmark images. Output: attention_heatmap.pdf/.png

Usage (run locally):
    python visualize_attention.py \
        --attention-dir ./attention_output/attention_maps \
        --image-dir ../data/synthetic \
        --output-dir ../../FontVlm/figs
"""
import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from PIL import Image
from scipy.ndimage import zoom

# Match paper style from gen_heatmap.py
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 10,
    "axes.linewidth": 0.8,
})

# Property display names and ordering
PROPERTY_ORDER = ["font_family", "font_size", "font_style"]
PROPERTY_LABELS = {
    "font_family": "Font Family",
    "font_size": "Font Size",
    "font_style": "Font Style",
}


def load_attention_data(attention_dir):
    """Load manifest and organize by property."""
    manifest_path = Path(attention_dir) / "manifest.json"
    with open(manifest_path) as f:
        manifest = json.load(f)

    # Group by property, maintaining order
    by_property = {}
    for entry in manifest:
        prop = entry["property"]
        if prop not in by_property:
            by_property[prop] = []
        by_property[prop].append(entry)

    return by_property


def resize_to_canvas(img_array, attn_norm, canvas_w, canvas_h):
    """Resize image + attention to fixed canvas, preserving aspect ratio.

    Scales the image to fill the canvas as much as possible, then center-crops
    (or center-pads minimally) so every panel is exactly canvas_w × canvas_h.
    """
    from PIL import Image as _Image

    h, w = img_array.shape[:2]
    # Scale to fill (cover) — crop the overflow
    scale = max(canvas_w / w, canvas_h / h)
    new_w, new_h = int(w * scale), int(h * scale)

    img_resized = np.array(
        _Image.fromarray(img_array).resize((new_w, new_h), _Image.LANCZOS)
    )
    attn_resized = zoom(attn_norm, (new_h / attn_norm.shape[0], new_w / attn_norm.shape[1]), order=3)

    # Center-crop to exact canvas size
    y0 = max(0, (new_h - canvas_h) // 2)
    x0 = max(0, (new_w - canvas_w) // 2)
    img_crop = img_resized[y0:y0 + canvas_h, x0:x0 + canvas_w]
    attn_crop = attn_resized[y0:y0 + canvas_h, x0:x0 + canvas_w]

    return img_crop, attn_crop


def create_heatmap_figure(attention_dir, image_dir, output_dir):
    """Create 2×3 attention heatmap figure."""
    attention_dir = Path(attention_dir)
    image_dir = Path(image_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_property = load_attention_data(attention_dir)

    for prop in PROPERTY_ORDER:
        count = len(by_property.get(prop, []))
        if count < 2:
            print(f"WARNING: Only {count} samples for {prop} (expected 2)")

    # --- First pass: load all panels ---
    panels = {}
    for col_idx, prop in enumerate(PROPERTY_ORDER):
        entries = by_property.get(prop, [])
        for row_idx in range(2):
            if row_idx >= len(entries):
                continue
            entry = entries[row_idx]

            npz_path = attention_dir / entry["npz_file"]
            data = np.load(npz_path, allow_pickle=True)
            attn_map = data["attention_map"]
            true_answer = str(data["true_answer"])
            pred_answer = str(data["pred_answer"])

            img_path = image_dir / entry["image_path"]
            img = Image.open(img_path).convert("RGB")
            img_array = np.array(img)
            img_h, img_w = img_array.shape[:2]

            scale_h = img_h / attn_map.shape[0]
            scale_w = img_w / attn_map.shape[1]
            attn_upscaled = zoom(attn_map, (scale_h, scale_w), order=3)

            attn_min, attn_max = attn_upscaled.min(), attn_upscaled.max()
            if attn_max > attn_min:
                attn_norm = (attn_upscaled - attn_min) / (attn_max - attn_min)
            else:
                attn_norm = np.zeros_like(attn_upscaled)

            panels[(row_idx, col_idx)] = {
                "img": img_array,
                "attn": attn_norm,
                "true": true_answer,
                "pred": pred_answer,
            }

    # --- Fixed canvas: all panels rendered at the same pixel size ---
    CANVAS_W, CANVAS_H = 400, 160  # ~2.5:1 landscape

    # Figure: 3 tight columns + thin colorbar, 2 rows
    cell_w_in = 2.2
    cell_h_in = cell_w_in * CANVAS_H / CANVAS_W
    fig_w = cell_w_in * 3 + 0.55  # 3 cols + cbar + margins
    fig_h = cell_h_in * 2 + 1.1   # 2 rows + titles/captions
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs = fig.add_gridspec(
        2, 4,
        width_ratios=[1, 1, 1, 0.04],
        wspace=0.08, hspace=0.40,
        left=0.01, right=0.92, top=0.90, bottom=0.06,
    )
    axes = np.empty((2, 3), dtype=object)
    for r in range(2):
        for c in range(3):
            axes[r, c] = fig.add_subplot(gs[r, c])

    for col_idx, prop in enumerate(PROPERTY_ORDER):
        for row_idx in range(2):
            ax = axes[row_idx, col_idx]

            if (row_idx, col_idx) not in panels:
                ax.axis("off")
                continue

            p = panels[(row_idx, col_idx)]
            img_c, attn_c = resize_to_canvas(p["img"], p["attn"], CANVAS_W, CANVAS_H)

            ax.imshow(img_c)
            ax.imshow(attn_c, cmap="jet", alpha=0.45,
                      extent=[0, CANVAS_W, CANVAS_H, 0])

            caption = f"True: {p['true']}  |  Pred: {p['pred']}"
            ax.set_xlabel(caption, fontsize=6.5, labelpad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_edgecolor("#999999")
                spine.set_linewidth(0.5)

            if row_idx == 0:
                ax.set_title(PROPERTY_LABELS[prop], fontsize=10, fontweight="bold", pad=4)

    # Colorbar
    cbar_ax = fig.add_subplot(gs[:, 3])
    norm = Normalize(vmin=0, vmax=1)
    sm = plt.cm.ScalarMappable(cmap="jet", norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_ticks([0, 0.5, 1.0])
    cbar.set_ticklabels(["Low", "Med", "High"])
    cbar.ax.tick_params(labelsize=7)
    cbar.set_label("Attention", fontsize=8)

    # Save
    pdf_path = output_dir / "attention_heatmap.pdf"
    png_path = output_dir / "attention_heatmap.png"
    fig.savefig(pdf_path, bbox_inches="tight", dpi=300)
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    plt.close(fig)

    print(f"Saved: {pdf_path}")
    print(f"Saved: {png_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize attention heatmaps")
    parser.add_argument(
        "--attention-dir",
        default="./attention_output/attention_maps",
        help="Directory containing .npz files and manifest.json",
    )
    parser.add_argument(
        "--image-dir",
        default="../data/synthetic",
        help="Directory containing benchmark images",
    )
    parser.add_argument(
        "--output-dir",
        default="../../FontVlm/figs",
        help="Output directory for figure",
    )
    args = parser.parse_args()

    create_heatmap_figure(args.attention_dir, args.image_dir, args.output_dir)
