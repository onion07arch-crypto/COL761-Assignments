import numpy as np
import faiss
import os
from multiprocessing.pool import ThreadPool


def to_float32_contiguous(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x.astype(np.float32, copy=False))


def build_ivf_index(xb, nlist, nprobe, seed=0):
    d = xb.shape[1]
    nlist = int(max(1, min(nlist, xb.shape[0])))

    quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)

    rng = np.random.default_rng(seed)
    train_size = min(xb.shape[0], max(50_000, 40 * nlist))
    train_ids = rng.choice(xb.shape[0], size=train_size, replace=False)

    index.train(xb[train_ids])
    index.add(xb)
    index.nprobe = int(max(1, min(nprobe, nlist)))
    return index


def aggregate_scores(index, xq, k, num_base, batch_size):
    num_threads = os.cpu_count() or 10
    # 1 OMP thread per call — ThreadPool provides real parallelism
    faiss.omp_set_num_threads(1)

    rank_weights = 1.0 / np.log2(np.arange(2, k + 2, dtype=np.float64))
    rank_ids = np.arange(k, dtype=np.int64)

    # Split queries into per-thread chunks
    chunks = [c for c in np.array_split(xq, num_threads) if len(c) > 0]

    def search_chunk(chunk):
        local = np.zeros(num_base, dtype=np.float64)
        chunk = np.ascontiguousarray(chunk, dtype=np.float32)
        for start in range(0, len(chunk), batch_size):
            end = min(start + batch_size, len(chunk))
            _, neighbors = index.search(chunk[start:end], k)
            valid = neighbors >= 0
            if not np.any(valid):
                continue
            cols = np.broadcast_to(rank_ids, neighbors.shape)[valid]
            flat_neighbors = neighbors[valid]
            local += np.bincount(flat_neighbors,
                                 weights=rank_weights[cols],
                                 minlength=num_base)
        return local

    with ThreadPool(processes=num_threads) as pool:
        partials = pool.map(search_chunk, chunks)

    return np.sum(partials, axis=0)


def rank_indices(scores, K):
    idx = np.arange(scores.shape[0], dtype=np.int64)
    order = np.lexsort((idx, -scores))
    return order[:K].astype(np.int64)


def finalize_answer(ans, N, K):
    ans = np.asarray(ans, dtype=np.int64).reshape(-1)
    ans = ans[(0 <= ans) & (ans < N)]
    seen = set()
    out = []
    for idx in ans:
        idx_int = int(idx)
        if idx_int not in seen:
            seen.add(idx_int)
            out.append(idx_int)
        if len(out) == K:
            break
    if len(out) < K:
        for idx in range(N):
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
            if len(out) == K:
                break
    return np.asarray(out, dtype=np.int64)


def solve(base_vectors, query_vectors, k, K, time_budget):
    xb = to_float32_contiguous(base_vectors)
    xq = to_float32_contiguous(query_vectors)

    N = xb.shape[0]
    if N == 0 or K <= 0:
        return np.zeros((0,), dtype=np.int64)

    k = int(min(max(k, 1), N))
    K = int(min(max(K, 1), N))

    faiss.omp_set_num_threads(os.cpu_count() or 4)

    if time_budget <= 20.0:
        nlist = int(np.clip(2.0 * np.sqrt(max(N, 1)), 128, 768))
        nprobe = min(8, nlist)
        batch_size = 2048
    elif time_budget <= 40.0:
        nlist = int(np.clip(3.0 * np.sqrt(max(N, 1)), 256, 1536))
        nprobe = min(12, nlist)
        batch_size = 4096
    else:
        nlist = int(np.clip(4.0 * np.sqrt(max(N, 1)), 256, 2048))
        nprobe = min(20, nlist)
        batch_size = 8192

    index = build_ivf_index(xb=xb, nlist=nlist, nprobe=nprobe, seed=0)

    scores = aggregate_scores(index=index, xq=xq, k=k,
                              num_base=N, batch_size=batch_size)

    ans = rank_indices(scores, K)
    return finalize_answer(ans, N, K)