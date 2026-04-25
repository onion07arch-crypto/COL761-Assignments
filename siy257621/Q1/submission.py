import os
import time
import numpy as np
import faiss

def _as_float32_contiguous(x):
    x = np.asarray(x)
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    return np.ascontiguousarray(x)


def _finalize_answer(indices, n_items, K):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)

    out = []
    seen = set()

    for idx in indices:
        idx = int(idx)
        if 0 <= idx < n_items and idx not in seen:
            seen.add(idx)
            out.append(idx)
            if len(out) == K:
                break

    if len(out) < K:
        for idx in range(n_items):
            if idx not in seen:
                seen.add(idx)
                out.append(idx)
                if len(out) == K:
                    break

    return np.asarray(out, dtype=np.int64)


def _rank_by_frequency(counts, K):
    idx = np.arange(counts.shape[0], dtype=np.int64)
    order = np.lexsort((idx, -counts))
    return order[:K].astype(np.int64)


def _centroid_fallback(xb, xq, K):
    n_items = xb.shape[0]
    if xq.shape[0] == 0:
        return np.arange(min(K, n_items), dtype=np.int64)

    centroid = np.ascontiguousarray(xq.mean(axis=0, keepdims=True).astype(np.float32))
    index = faiss.IndexFlatL2(xb.shape[1])
    index.add(xb)
    _, ids = index.search(centroid, min(K, n_items))
    return _finalize_answer(ids.reshape(-1), n_items, K)


def _search_and_count(index, xq, k, n_items, batch_size, deadline):
    counts = np.zeros(n_items, dtype=np.int64)
    q_count = xq.shape[0]

    if q_count == 0:
        return counts, 0
    rng = np.random.default_rng(12345)
    if q_count > batch_size:
        order = rng.permutation(q_count)
    else:
        order = np.arange(q_count)

    processed = 0

    for start in range(0, q_count, batch_size):
        if time.perf_counter() >= deadline:
            break

        end = min(start + batch_size, q_count)
        batch_ids = order[start:end]
        q_batch = np.ascontiguousarray(xq[batch_ids])

        _, neighbors = index.search(q_batch, k)
        neighbors = neighbors.reshape(-1)
        neighbors = neighbors[neighbors >= 0]

        if neighbors.size > 0:
            counts += np.bincount(neighbors, minlength=n_items)

        processed += end - start

    return counts, processed


def _use_exact_search(n_items, q_count, dim, time_budget):
    if n_items <= 25000:
        return True

    work = float(n_items) * float(q_count) * float(dim)

    if time_budget <= 22.0:
        return work <= 2.0e9
    if time_budget <= 45.0:
        return work <= 5.0e9
    return work <= 10.0e9


def _choose_ivf_params(n_items, dim, time_budget):
    root_n = np.sqrt(max(n_items, 1))

    if time_budget <= 22.0:
        nlist = int(np.clip(4.0 * root_n, 128, 2048))
        nprobe = 8
        train_cap = 60000
        batch_size = 1024 if dim <= 256 else 512
    elif time_budget <= 45.0:
        nlist = int(np.clip(6.0 * root_n, 256, 4096))
        nprobe = 16
        train_cap = 120000
        batch_size = 2048 if dim <= 256 else 1024
    else:
        nlist = int(np.clip(8.0 * root_n, 512, 8192))
        nprobe = 32
        train_cap = 240000
        batch_size = 4096 if dim <= 256 else 2048

    nlist = max(1, min(nlist, n_items))
    nprobe = max(1, min(nprobe, nlist))

    train_size = min(n_items, max(10000, 30 * nlist))
    train_size = min(train_size, train_cap)
    train_size = max(min(n_items, train_size), min(n_items, nlist))

    return nlist, nprobe, train_size, batch_size


def _build_ivf_index(xb, nlist, nprobe, train_size, seed=0):
    n_items, dim = xb.shape

    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_L2)

    rng = np.random.default_rng(seed)
    if train_size < n_items:
        train_ids = rng.choice(n_items, size=train_size, replace=False)
        train_x = np.ascontiguousarray(xb[train_ids])
    else:
        train_x = xb

    index.train(train_x)
    index.add(xb)
    index.nprobe = nprobe
    return index


def solve(base_vectors, query_vectors, k, K, time_budget):
    start_time = time.perf_counter()
    safe_budget = max(float(time_budget), 1.0)
    deadline = start_time + 0.92 * safe_budget

    xb = _as_float32_contiguous(base_vectors)
    xq = _as_float32_contiguous(query_vectors)

    n_items = xb.shape[0]
    if n_items == 0 or K <= 0:
        return np.zeros((0,), dtype=np.int64)

    K = int(min(max(int(K), 1), n_items))
    k = int(min(max(int(k), 1), n_items))

    q_count = xq.shape[0]
    dim = xb.shape[1]

    if q_count == 0:
        return np.arange(K, dtype=np.int64)

    try:
        if _use_exact_search(n_items, q_count, dim, safe_budget):
            index = faiss.IndexFlatL2(dim)
            index.add(xb)
            batch_size = 2048 if dim <= 256 else 1024
        else:
            nlist, nprobe, train_size, batch_size = _choose_ivf_params(
                n_items=n_items,
                dim=dim,
                time_budget=safe_budget,
            )

            if time.perf_counter() >= deadline:
                return _centroid_fallback(xb, xq, K)

            index = _build_ivf_index(
                xb=xb,
                nlist=nlist,
                nprobe=nprobe,
                train_size=train_size,
                seed=0,
            )

        counts, processed = _search_and_count(
            index=index,
            xq=xq,
            k=k,
            n_items=n_items,
            batch_size=batch_size,
            deadline=deadline,
        )

        if processed == 0 or counts.sum() == 0:
            ans = _centroid_fallback(xb, xq, K)
        else:
            ans = _rank_by_frequency(counts, K)

        return _finalize_answer(ans, n_items, K)

    except Exception:
        return _finalize_answer(np.arange(K, dtype=np.int64), n_items, K)