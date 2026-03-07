#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SOURCE_DIR="${1:-$ROOT_DIR/../FontVlm}"

mkdir -p "$ROOT_DIR/assets/project" "$ROOT_DIR/paper"

cp "$SOURCE_DIR/main.pdf" "$ROOT_DIR/paper/reading-not-seeing.pdf"

for filename in \
  attention_heatmap.png \
  dataset_stats.png \
  finetuning_bar.png \
  heatmap.png \
  hierarchy_concept.png \
  pipeline_concept.png \
  pipeline_concept_1.png \
  radar_chart.png \
  resolution_robustness.png \
  sample_gallery_concept.png \
  stroop_bar.png \
  teaser_concept.png
do
  cp "$SOURCE_DIR/figs/$filename" "$ROOT_DIR/assets/project/$filename"
done

printf 'Synced paper assets from %s\n' "$SOURCE_DIR"
