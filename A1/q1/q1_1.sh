#!/bin/bash


# Arguments

APRIORI=$1
FPGROWTH=$2
DATASET=$3
OUTDIR=$4


# Checks

if [ "$#" -ne 4 ]; then
  echo "Usage: bash q1_1.sh <apriori_path> <fpgrowth_path> <dataset_path> <output_dir>"
  exit 1
fi

mkdir -p "$OUTDIR"

# Support thresholds
SUPPORTS=(5 10 25 50 90)

# Files to store runtimes
APR_TIME="$OUTDIR/apriori_times.txt"
FPG_TIME="$OUTDIR/fpgrowth_times.txt"

echo -n "" > "$APR_TIME"
echo -n "" > "$FPG_TIME"


# Run Apriori

for s in "${SUPPORTS[@]}"; do
  OUT="$OUTDIR/ap$s"
  mkdir -p "$OUT"

  START=$(date +%s.%N)
  "$APRIORI" -s"$s" "$DATASET" "$OUT/result.txt" > /dev/null 2>&1
  END=$(date +%s.%N)

  TIME=$(echo "$END - $START" | bc)
  echo "$s $TIME" >> "$APR_TIME"
done


# Run FP-Growth

for s in "${SUPPORTS[@]}"; do
  OUT="$OUTDIR/fp$s"
  mkdir -p "$OUT"

  START=$(date +%s.%N)
  "$FPGROWTH" -s"$s" "$DATASET" "$OUT/result.txt" > /dev/null 2>&1
  END=$(date +%s.%N)

  TIME=$(echo "$END - $START" | bc)
  echo "$s $TIME" >> "$FPG_TIME"
done

# Plot (Python)

python3 << EOF
import matplotlib.pyplot as plt

# Read times
ap_s, ap_t = zip(*[(int(a), float(b)) for a, b in
                   (line.split() for line in open("$APR_TIME"))])
fp_s, fp_t = zip(*[(int(a), float(b)) for a, b in
                   (line.split() for line in open("$FPG_TIME"))])

plt.plot(ap_s, ap_t, marker='o', label='Apriori')
plt.plot(fp_s, fp_t, marker='o', label='FP-Growth')

plt.xlabel("Support Threshold (%)")
plt.ylabel("Runtime (seconds)")
plt.title("Apriori vs FP-Growth Runtime Comparison")
plt.legend()
plt.grid(True)

plt.savefig("$OUTDIR/plot.png")
EOF

echo "Done. Outputs saved in $OUTDIR"
