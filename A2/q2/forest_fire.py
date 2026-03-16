import sys
import os
import heapq
import random
import time
from collections import defaultdict, deque

# ── I/O helpers ────────────────────────────────────────────────────────────────

def load_graph(path):
    adj   = defaultdict(list)
    edges = []
    probs = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 3:
                u, v, p = int(parts[0]), int(parts[1]), float(parts[2])
            elif len(parts) == 2:
                u, v, p = int(parts[0]), int(parts[1]), 1.0
            else:
                continue
            adj[u].append((v, p))
            edges.append((u, v))
            probs[(u, v)] = p
    return adj, edges, probs


def load_seeds(path):
    seeds = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                seeds.append(int(line.split()[0]))
    return seeds


def write_output(path, routes):
    with open(path, 'w') as f:
        for u, v in routes:
            f.write(f"{u} {v}\n")

# ── Fire simulation ─────────────────────────────────────────────────────────────

def simulate_once(adj, seeds, blocked_set, hops, probs):
    burned    = set(seeds)
    queue     = deque(seeds)
    hop_dist  = {s: 0 for s in seeds}   # only used when hops != -1

    while queue:
        u = queue.popleft()
        d_u = hop_dist.get(u, 0) if hops != -1 else 0

        for (v, p) in adj[u]:
            if v in burned:
                continue
            if (u, v) in blocked_set:
                continue
            if hops != -1 and d_u + 1 > hops:
                continue
            if random.random() < p:
                burned.add(v)
                if hops != -1:
                    hop_dist[v] = d_u + 1
                queue.append(v)

    return len(burned)


def estimate_sigma(adj, seeds, blocked_set, hops, probs, n_sim):
    total = 0
    for _ in range(n_sim):
        total += simulate_once(adj, seeds, blocked_set, hops, probs)
    return total / n_sim

# ── CELF greedy ─────────────────────────────────────────────────────────────────

def celf_greedy(adj, edges, probs, seeds, k, n_sim, hops, output_path):
    blocked = set()
    selected = []

    # Compute baseline
    sigma_empty = estimate_sigma(adj, seeds, blocked, hops, probs, n_sim)
    print(f"[INFO] sigma(null) = {sigma_empty:.4f}", flush=True)

    # --- Initial marginal gain priority queue (max-heap via negation) ---
    # heap entries: (-gain, edge, iteration_added)
    heap = []
    current_sigma = sigma_empty

    # Warm-start: compute gain for every candidate edge
    n_sim_init = n_sim

    print(f"[INFO] |E|={len(edges)}, init sims={n_sim_init}, full sims={n_sim}", flush=True)

    for (u, v) in edges:
        b = blocked | {(u, v)}
        gain = current_sigma - estimate_sigma(adj, seeds, b, hops, probs, n_sim_init)
        heapq.heappush(heap, (-gain, 0, u, v))   # 0 = iteration when gain was computed

    iteration = 0
    while len(selected) < k and heap:
        while True:
            neg_gain, iter_computed, u, v = heapq.heappop(heap)
            if (u, v) in blocked:
                continue
            if iter_computed == iteration:
                # Gain is current — accept
                break
            else:
                # Recompute marginal gain
                b = blocked | {(u, v)}
                gain = current_sigma - estimate_sigma(adj, seeds, b, hops, probs, n_sim)
                heapq.heappush(heap, (-gain, iteration, u, v))
                # Loop: the heap will now return the freshest best candidate

        # Select edge (u, v)
        blocked.add((u, v))
        selected.append((u, v))
        iteration += 1
        current_sigma = estimate_sigma(adj, seeds, blocked, hops, probs, n_sim)
        print(
            f"[{iteration}/{k}] blocked ({u},{v}), gain={-neg_gain:.4f}, "
            f"sigma(R)={current_sigma:.4f}",
            flush=True
        )

        # Write partial output after every selection (partial credit on timeout)
        write_output(output_path, selected)

    reduction = (sigma_empty - current_sigma) / sigma_empty if sigma_empty > 0 else 0
    print(f"[INFO] Final sigma(R)={current_sigma:.4f}, reduction ratio={reduction:.4f}", flush=True)
    return selected

# ── Entry point ─────────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) != 7:
        print("Usage: python3 forest_fire.py <graph> <seed_set> <output> <k> <n_sim> <hops>")
        sys.exit(1)

    graph_path  = sys.argv[1]
    seed_path   = sys.argv[2]
    output_path = sys.argv[3]
    k           = int(sys.argv[4])
    n_sim       = int(sys.argv[5])
    hops        = int(sys.argv[6])   # -1 means unlimited

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    print(f"[INFO] Loading graph from {graph_path}", flush=True)
    adj, edges, probs = load_graph(graph_path)
    seeds = load_seeds(seed_path)
    print(f"[INFO] |V_adj|={len(adj)}, |E|={len(edges)}, |seeds|={len(seeds)}, k={k}, hops={hops}", flush=True)

    # Remove edges that cannot possibly be on any fire-spread path
    # (simple pre-filter: keep only edges reachable from seeds within hops BFS)
    candidate_edges = prefilter_edges(adj, edges, seeds, hops)
    print(f"[INFO] Candidate edges after pre-filter: {len(candidate_edges)}", flush=True)

    selected = celf_greedy(adj, candidate_edges, probs, seeds, k, n_sim, hops, output_path)
    write_output(output_path, selected)
    print("[INFO] Done.", flush=True)


def prefilter_edges(adj, edges, seeds, hops):
    if hops == -1:
        reachable = bfs_reachable(adj, seeds, limit=None)
    else:
        reachable = bfs_reachable(adj, seeds, limit=hops - 1)  # u must be within hops-1 of seeds

    return [(u, v) for (u, v) in edges if u in reachable]


def bfs_reachable(adj, seeds, limit):
    visited = set(seeds)
    queue   = deque((s, 0) for s in seeds)
    while queue:
        u, d = queue.popleft()
        if limit is not None and d >= limit:
            continue
        for (v, _) in adj[u]:
            if v not in visited:
                visited.add(v)
                queue.append((v, d + 1))
    return visited


if __name__ == "__main__":
    main()
