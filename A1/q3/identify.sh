#!/bin/bash
if [ $# -ne 2 ]; then
    echo "Usage: $0 <path_graph_dataset> <path_discriminative_subgraphs>"
    exit 1
fi
python3 identify_features.py "$1" "$2"