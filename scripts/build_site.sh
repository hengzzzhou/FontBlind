#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUTPUT_DIR="${1:-$ROOT_DIR/site-dist}"

rm -rf "$OUTPUT_DIR"
mkdir -p "$OUTPUT_DIR/assets" "$OUTPUT_DIR/paper" "$OUTPUT_DIR/fontbench/data/synthetic"

cp "$ROOT_DIR/index.html" "$ROOT_DIR/styles.css" "$ROOT_DIR/script.js" \
  "$ROOT_DIR/README.md" "$ROOT_DIR/LICENSE" "$OUTPUT_DIR/"
cp -R "$ROOT_DIR/assets/project" "$OUTPUT_DIR/assets/"
cp -R "$ROOT_DIR/paper" "$OUTPUT_DIR/"
cp "$ROOT_DIR/fontbench/data/synthetic/metadata.json" \
  "$OUTPUT_DIR/fontbench/data/synthetic/metadata.json"

touch "$OUTPUT_DIR/.nojekyll"

printf 'Built GitHub Pages artifact at %s\n' "$OUTPUT_DIR"
