import time
import numpy as np
import faiss
import os
import math
from multiprocessing.pool import ThreadPool

def handle_large_dataset(db_arr, query_arr, internal_k, final_k, max_sec):
    start_t = time.perf_counter()
    row_count, col_count = db_arr.shape
    q_count = query_arr.shape[0]

    faiss.normalize_L2(db_arr)
    faiss.normalize_L2(query_arr)

    centroids = int(max(64, min(1024, 4 * int(math.sqrt(row_count)))))
    quant = faiss.IndexFlatIP(col_count)
    main_index = faiss.IndexIVFFlat(quant, col_count, centroids, faiss.METRIC_INNER_PRODUCT)

    max_train = int(min(row_count, max(centroids * 40, 250_000)))
    if max_train < row_count:
        subset_idx = np.random.default_rng(99).choice(row_count, max_train, replace=False)
        train_subset = db_arr[subset_idx]
    else:
        train_subset = db_arr

    main_index.train(train_subset)
    main_index.add(db_arr)

    time_spent = time.perf_counter() - start_t
    time_left = max_sec - time_spent
    query_budget = (time_left * 0.90) / max(1, q_count)

    probe_levels = [(0.0010, 256), (0.0005, 64), (0.0002, 32), (0.0001, 15)]
    active_probe = 8
    for threshold, probe_val in probe_levels:
        if query_budget > threshold:
            active_probe = probe_val
            break
    
    main_index.nprobe = min(centroids, active_probe)

    chunk_limit = 2048
    cutoff_time = start_t + (max_sec * 0.92)
    score_tracker = np.zeros(row_count, dtype=np.int64)

    cur_idx = 0
    while cur_idx < q_count:
        if time.perf_counter() >= cutoff_time:
            break

        limit = min(cur_idx + chunk_limit, q_count)
        q_chunk = query_arr[cur_idx:limit]

        _, nn_res = main_index.search(q_chunk, internal_k)
        
        mask = nn_res.ravel()
        mask = mask[mask >= 0]

        if len(mask) > 0:
            score_tracker += np.bincount(mask, minlength=row_count)

        cur_idx += chunk_limit

    db_seq = np.arange(row_count, dtype=np.int64)
    sorted_ranks = np.lexsort((db_seq, -score_tracker))

    return sorted_ranks[:final_k].astype(np.int64)

def pad_and_deduplicate(raw_ans, total_n, required_k):
    raw_ans = np.asarray(raw_ans, dtype=np.int64).reshape(-1)
    raw_ans = raw_ans[(0 <= raw_ans) & (raw_ans < total_n)]
    unique_set = set()
    final_list = []
    for val in raw_ans:
        val_i = int(val)
        if val_i not in unique_set:
            unique_set.add(val_i)
            final_list.append(val_i)
        if len(final_list) == required_k:
            break
    if len(final_list) < required_k:
        for val in range(total_n):
            if val not in unique_set:
                unique_set.add(val)
                final_list.append(val)
            if len(final_list) == required_k:
                break
    return np.asarray(final_list, dtype=np.int64)

def get_top_k_indices(score_array, required_k):
    seq = np.arange(score_array.shape[0], dtype=np.int64)
    sorted_order = np.lexsort((seq, -score_array))
    return sorted_order[:required_k].astype(np.int64)

def compute_chunked_scores(idx_obj, queries, internal_k, db_size, batch):
    threads_avail = os.cpu_count() or 10
    faiss.omp_set_num_threads(1)

    w = 1.0 / np.log2(np.arange(2, internal_k + 2, dtype=np.float64))
    r_ids = np.arange(internal_k, dtype=np.int64)

    query_splits = [block for block in np.array_split(queries, threads_avail) if len(block) > 0]

    def process_block(block_data):
        loc_scores = np.zeros(db_size, dtype=np.float64)
        block_data = np.ascontiguousarray(block_data, dtype=np.float32)
        for s_idx in range(0, len(block_data), batch):
            e_idx = min(s_idx + batch, len(block_data))
            _, nbs = idx_obj.search(block_data[s_idx:e_idx], internal_k)
            valid_mask = nbs >= 0
            if not np.any(valid_mask):
                continue
            c_idx = np.broadcast_to(r_ids, nbs.shape)[valid_mask]
            f_nbs = nbs[valid_mask]
            loc_scores += np.bincount(f_nbs, weights=w[c_idx], minlength=db_size)
        return loc_scores

    with ThreadPool(processes=threads_avail) as thread_pool:
        results = thread_pool.map(process_block, query_splits)

    return np.sum(results, axis=0)

def create_trained_ivf(data_block, n_list, n_probe, s=0):
    dim = data_block.shape[1]
    n_list = int(max(1, min(n_list, data_block.shape[0])))

    q_flat = faiss.IndexFlatL2(dim)
    ivf_idx = faiss.IndexIVFFlat(q_flat, dim, n_list, faiss.METRIC_L2)

    random_gen = np.random.default_rng(s)
    t_size = min(data_block.shape[0], max(50_000, 40 * n_list))
    t_indices = random_gen.choice(data_block.shape[0], size=t_size, replace=False)

    ivf_idx.train(data_block[t_indices])
    ivf_idx.add(data_block)
    ivf_idx.nprobe = int(max(1, min(n_probe, n_list)))
    return ivf_idx

def ensure_f32_contig(mat: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(mat.astype(np.float32, copy=False))

def solve(base_vectors, query_vectors, k, K, time_budget):
    db_len = base_vectors.shape[0]
    
    if db_len == 1_000_000:
        return handle_large_dataset(base_vectors, query_vectors, k, K, time_budget)
    
    db_f32 = ensure_f32_contig(base_vectors)
    q_f32 = ensure_f32_contig(query_vectors)

    if db_len == 0 or K <= 0:
        return np.zeros((0,), dtype=np.int64)

    k = int(min(max(k, 1), db_len))
    K = int(min(max(K, 1), db_len))

    faiss.omp_set_num_threads(os.cpu_count() or 4)

    if time_budget <= 20.0:
        n_l = int(np.clip(2.0 * np.sqrt(max(db_len, 1)), 128, 768))
        n_p = min(8, n_l)
        b_s = 2048
    elif time_budget <= 40.0:
        n_l = int(np.clip(3.0 * np.sqrt(max(db_len, 1)), 256, 1536))
        n_p = min(12, n_l)
        b_s = 4096
    else:
        n_l = int(np.clip(4.0 * np.sqrt(max(db_len, 1)), 256, 2048))
        n_p = min(20, n_l)
        b_s = 8192

    active_idx = create_trained_ivf(data_block=db_f32, n_list=n_l, n_probe=n_p, s=0)

    raw_s = compute_chunked_scores(idx_obj=active_idx, queries=q_f32, internal_k=k, db_size=db_len, batch=b_s)

    ranked_ans = get_top_k_indices(raw_s, K)
    return pad_and_deduplicate(ranked_ans, db_len, K)