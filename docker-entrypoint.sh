#!/usr/bin/env bash
set -Eeo pipefail
echo "-- Starting MONAI Segmentation Bundle..."
conda run -n mercure-monaisegment python seg_app -i $MERCURE_IN_DIR -o $MERCURE_OUT_DIR -m model.ts
echo "-- Done."