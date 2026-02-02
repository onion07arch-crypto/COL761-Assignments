#!/bin/bash

if [ $# -ne 5 ]; then
    echo "Usage: $0 <gspan_exe> <fsg_exe> <gaston_exe> <dataset> <output_dir>"
    exit 1
fi

GSPAN_EXE="$1"
FSG_EXE="$2"
GASTON_EXE="$3"
DATASET="$4"
OUTPUT_DIR="$5"

python3 run_q2.py "$GSPAN_EXE" "$FSG_EXE" "$GASTON_EXE" "$DATASET" "$OUTPUT_DIR"