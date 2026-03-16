#!/bin/bash
# forest_fire.sh
# Usage: bash forest_fire.sh <graph_path> <seed_set_path> <output_path> <k> <n_random_instances> <hops>

GRAPH_PATH=$1
SEED_PATH=$2
OUTPUT_PATH=$3
K=$4
N_RANDOM=$5
HOPS=$6

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "$#" -lt 5 ] || [ "$#" -gt 6 ]; then
    echo "Usage: bash forest_fire.sh <graph_file> <seed_file> <output_path> <k> <num_sim> [hops]"
    echo ""
    echo "  hops  (optional) — limit fire spread to this many hops from the seed set."
    echo "                     Pass -1 or omit for unlimited spread (default)."
    exit 1
fi

python3 "$SCRIPT_DIR/forest_fire.py" \
    "$GRAPH_PATH" "$SEED_PATH" "$OUTPUT_PATH" "$K" "$N_RANDOM" "$HOPS"
