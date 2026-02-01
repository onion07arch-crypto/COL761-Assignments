#!/bin/bash



if [ "$#" -ne 2 ]; then
  echo "Usage: bash q1_2.sh <universal_itemset_size> <num_transactions>"
  exit 1
fi

ITEMSET_SIZE=$1
NUM_TRANSACTIONS=$2

python3 generate_dataset.py "$ITEMSET_SIZE" "$NUM_TRANSACTIONS"
