import sys
import random


# Argument check

if len(sys.argv) != 3:
    print("Usage: python generate_dataset.py <num_items> <num_transactions>")
    sys.exit(1)

NUM_ITEMS = int(sys.argv[1])
NUM_TRANS = int(sys.argv[2])

random.seed(42)


# Create universal itemset

items = [f"i{i}" for i in range(NUM_ITEMS)]

transactions = set()


# Generate transactions

while len(transactions) < NUM_TRANS:
    # Long transactions → slow Apriori
    txn_len = random.randint(NUM_ITEMS // 4, NUM_ITEMS // 2)

    # Bias frequent items
    frequent_part = random.sample(items[:NUM_ITEMS // 3], txn_len // 2)
    random_part = random.sample(items, txn_len - len(frequent_part))

    txn = set(frequent_part + random_part)
    transactions.add(" ".join(sorted(txn)))


# Write dataset

with open("generated_transactions.dat", "w") as f:
    for t in transactions:
        f.write(t + "\n")

print(f"Generated dataset with {len(transactions)} transactions")
