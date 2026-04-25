import argparse
import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.utils import negative_sampling

from checkpoint_utils import save_checkpoint
from load_dataset import load_dataset
from model_zoo import LinkGraphSAGE, NodeGraphSAGE


NODE_CONFIG = {
    "A": {
        "hidden_channels": 256,
        "num_layers": 3,
        "dropout": 0.45,
        "lr": 1e-3,
        "weight_decay": 5e-4,
        "max_epochs": 350,
        "patience": 50,
    },
    "B": {
        "hidden_channels": 128,
        "num_layers": 3,
        "dropout": 0.35,
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "max_epochs": 300,
        "patience": 45,
    },
}


LINK_CONFIG = {
    "hidden_channels": 128,
    "num_layers": 2,
    "dropout": 0.25,
    "lr": 8e-4,
    "weight_decay": 1e-5,
    "max_epochs": 220,
    "patience": 35,
    "edge_batch_size": 65536,
    "decode_batch_size": 262144,
    "eval_every": 2,
}


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(obj, device):
    """
    Move PyG Data objects or simple dataset wrappers to device.
    """
    if torch.is_tensor(obj):
        return obj.to(device)

    if hasattr(obj, "to") and callable(obj.to):
        try:
            return obj.to(device)
        except Exception:
            pass

    if hasattr(obj, "__dict__"):
        for name, value in vars(obj).items():
            if torch.is_tensor(value):
                setattr(obj, name, value.to(device))

    return obj


def binary_auc(y_true: torch.Tensor, y_score: torch.Tensor) -> float:
    """
    Small dependency-free ROC-AUC implementation.
    """
    y_true_np = y_true.detach().view(-1).cpu().numpy().astype(np.int64)
    y_score_np = y_score.detach().view(-1).cpu().numpy().astype(np.float64)

    pos_mask = y_true_np == 1
    neg_mask = y_true_np == 0

    n_pos = int(pos_mask.sum())
    n_neg = int(neg_mask.sum())

    if n_pos == 0 or n_neg == 0:
        return 0.5

    order = np.argsort(y_score_np, kind="mergesort")
    sorted_scores = y_score_np[order]

    ranks = np.empty_like(sorted_scores, dtype=np.float64)

    start = 0
    while start < sorted_scores.shape[0]:
        end = start + 1
        while end < sorted_scores.shape[0] and sorted_scores[end] == sorted_scores[start]:
            end += 1

        avg_rank = 0.5 * (start + end - 1) + 1.0
        ranks[start:end] = avg_rank
        start = end

    full_ranks = np.empty_like(ranks)
    full_ranks[order] = ranks

    pos_rank_sum = float(full_ranks[pos_mask].sum())
    auc = (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return float(auc)


def hits_at_k(pos_scores: torch.Tensor, neg_scores: torch.Tensor, k: int = 50) -> float:
    """
    Hits@k for grouped hard negatives.
    pos_scores: [P]
    neg_scores: [P, num_neg]
    """
    pos_scores = pos_scores.detach().view(-1).cpu()
    neg_scores = neg_scores.detach().cpu()

    ranks = 1 + (neg_scores > pos_scores.unsqueeze(1)).sum(dim=1)
    return float((ranks <= k).float().mean().item())


def get_node_task_tensors(data):
    x = data.x.float()
    edge_index = data.edge_index.long()
    y = data.y
    labeled_nodes = data.labeled_nodes.long()
    train_mask = data.train_mask.bool()
    val_mask = data.val_mask.bool()

    return x, edge_index, y, labeled_nodes, train_mask, val_mask


@torch.no_grad()
def evaluate_node_model(model, data, dataset_name: str, split_name: str) -> float:
    model.eval()

    x, edge_index, y, labeled_nodes, train_mask, val_mask = get_node_task_tensors(data)

    mask = train_mask if split_name == "train" else val_mask
    node_idx = labeled_nodes[mask]

    logits = model(x, edge_index)

    if dataset_name == "A":
        pred = logits[node_idx].argmax(dim=-1)
        target = y[mask].view(-1).long()
        return float((pred == target).float().mean().item())

    if dataset_name == "B":
        score = logits[node_idx].view(-1)
        target = y[mask].view(-1).float()
        return binary_auc(target, score)

    raise ValueError(f"Unsupported node dataset: {dataset_name}")


def train_node_task(data, dataset_name: str, device: torch.device):
    x, edge_index, y, labeled_nodes, train_mask, val_mask = get_node_task_tensors(data)
    cfg = NODE_CONFIG[dataset_name]

    if dataset_name == "A":
        out_channels = int(y.max().item()) + 1
    else:
        out_channels = 1

    model = NodeGraphSAGE(
        in_channels=x.size(-1),
        hidden_channels=cfg["hidden_channels"],
        out_channels=out_channels,
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        input_norm=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=10,
    )

    train_node_idx = labeled_nodes[train_mask]

    if dataset_name == "B":
        train_targets = y[train_mask].view(-1).float()
        pos_count = float(train_targets.sum().item())
        neg_count = float(train_targets.numel() - pos_count)

        pos_weight = torch.tensor(
            neg_count / max(pos_count, 1.0),
            device=device,
            dtype=torch.float32,
        )

        criterion_b = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    else:
        criterion_b = None

    best_metric = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_left = cfg["patience"]

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        logits = model(x, edge_index)

        if dataset_name == "A":
            target = y[train_mask].view(-1).long()
            loss = F.cross_entropy(logits[train_node_idx], target)
        else:
            target = y[train_mask].view(-1).float()
            train_score = logits[train_node_idx].view(-1)
            loss = criterion_b(train_score, target)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        val_metric = evaluate_node_model(model, data, dataset_name, "val")
        scheduler.step(val_metric)

        if val_metric > best_metric:
            best_metric = val_metric
            best_state = copy.deepcopy(model.state_dict())
            patience_left = cfg["patience"]
        else:
            patience_left -= 1

        if epoch % 10 == 0:
            print(
                f"epoch={epoch:04d} "
                f"loss={loss.item():.6f} "
                f"val_metric={val_metric:.6f} "
                f"best={best_metric:.6f}"
            )

        if patience_left <= 0:
            break

    model.load_state_dict(best_state)
    model.eval()

    return model, best_metric


def _flatten_edge_pairs(edge_pairs: torch.Tensor) -> torch.Tensor:
    """
    Return edges as [E, 2].
    """
    edge_pairs = edge_pairs.long()

    if edge_pairs.dim() == 3:
        return edge_pairs.view(-1, 2).contiguous()

    if edge_pairs.dim() == 2 and edge_pairs.size(0) == 2:
        return edge_pairs.t().contiguous()

    if edge_pairs.dim() == 2 and edge_pairs.size(1) == 2:
        return edge_pairs.contiguous()

    raise ValueError(f"Invalid edge-pair shape: {tuple(edge_pairs.shape)}")


def get_link_task_tensors(data):
    x = data.x.float()
    edge_index = data.edge_index.long()

    train_pos = _flatten_edge_pairs(data.train_pos)
    valid_pos = _flatten_edge_pairs(data.valid_pos)

    train_neg = None
    if hasattr(data, "train_neg"):
        train_neg = _flatten_edge_pairs(data.train_neg)

    valid_neg = data.valid_neg.long()
    if valid_neg.dim() != 3:
        raise ValueError("valid_neg must have shape [P, K, 2] for Hits@50 evaluation.")

    num_nodes = int(x.size(0))

    return x, edge_index, train_pos, train_neg, valid_pos, valid_neg, num_nodes


def sample_edges(edge_pairs: torch.Tensor, batch_size: int) -> torch.Tensor:
    n_edges = edge_pairs.size(0)
    if n_edges <= batch_size:
        return edge_pairs

    idx = torch.randint(
        low=0,
        high=n_edges,
        size=(batch_size,),
        device=edge_pairs.device,
    )

    return edge_pairs[idx]


def decode_edges_in_batches(model, z, edge_pairs, batch_size):
    edge_pairs = _flatten_edge_pairs(edge_pairs)

    scores = []
    for start in range(0, edge_pairs.size(0), batch_size):
        end = min(start + batch_size, edge_pairs.size(0))
        score = model.decode(z, edge_pairs[start:end])
        scores.append(score)

    return torch.cat(scores, dim=0)


@torch.no_grad()
def evaluate_link_model(model, data) -> float:
    model.eval()

    cfg = LINK_CONFIG
    x, edge_index, _, _, valid_pos, valid_neg, _ = get_link_task_tensors(data)

    z = model.encode(x, edge_index)

    pos_scores = decode_edges_in_batches(
        model=model,
        z=z,
        edge_pairs=valid_pos,
        batch_size=cfg["decode_batch_size"],
    )

    P, K, _ = valid_neg.shape

    neg_scores = decode_edges_in_batches(
        model=model,
        z=z,
        edge_pairs=valid_neg.view(P * K, 2),
        batch_size=cfg["decode_batch_size"],
    ).view(P, K)

    return hits_at_k(pos_scores, neg_scores, k=50)


def train_link_task(data, device: torch.device):
    x, edge_index, train_pos, train_neg, _, _, num_nodes = get_link_task_tensors(data)

    cfg = LINK_CONFIG

    model = LinkGraphSAGE(
        in_channels=x.size(-1),
        hidden_channels=cfg["hidden_channels"],
        num_layers=cfg["num_layers"],
        dropout=cfg["dropout"],
        input_norm=True,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
    )

    best_metric = float("-inf")
    best_state = copy.deepcopy(model.state_dict())
    patience_left = cfg["patience"]

    edge_batch_size = cfg["edge_batch_size"]

    for epoch in range(1, cfg["max_epochs"] + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        z = model.encode(x, edge_index)

        pos_edges = sample_edges(train_pos, edge_batch_size)

        if train_neg is not None and train_neg.size(0) > 0:
            neg_edges = sample_edges(train_neg, pos_edges.size(0))
        else:
            neg_edges = negative_sampling(
                edge_index=edge_index,
                num_nodes=num_nodes,
                num_neg_samples=pos_edges.size(0),
                method="sparse",
            ).t().contiguous()

        pos_scores = model.decode(z, pos_edges)
        neg_scores = model.decode(z, neg_edges)

        bce_pos = F.binary_cross_entropy_with_logits(
            pos_scores,
            torch.ones_like(pos_scores),
        )
        bce_neg = F.binary_cross_entropy_with_logits(
            neg_scores,
            torch.zeros_like(neg_scores),
        )

        pair_count = min(pos_scores.numel(), neg_scores.numel())
        rank_loss = F.softplus(neg_scores[:pair_count] - pos_scores[:pair_count]).mean()

        loss = bce_pos + bce_neg + 0.5 * rank_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
        optimizer.step()

        if epoch % cfg["eval_every"] == 0:
            val_metric = evaluate_link_model(model, data)
            scheduler.step(val_metric)

            if val_metric > best_metric:
                best_metric = val_metric
                best_state = copy.deepcopy(model.state_dict())
                patience_left = cfg["patience"]
            else:
                patience_left -= 1

            print(
                f"epoch={epoch:04d} "
                f"loss={loss.item():.6f} "
                f"val_hits50={val_metric:.6f} "
                f"best={best_metric:.6f}"
            )

            if patience_left <= 0:
                break

    model.load_state_dict(best_state)
    model.eval()

    return model, best_metric


def main():
    parser = argparse.ArgumentParser(description="Train COL761 Assignment 3 models.")
    parser.add_argument("--dataset", choices=["A", "B", "C"], required=True)
    parser.add_argument("--task", choices=["node", "link"], required=True)
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    print(f"Using device: {device}")

    if args.dataset in {"A", "B"}:
        dataset = load_dataset(args.dataset, args.data_dir)
        data = dataset[0]
    else:
        data = load_dataset("C", args.data_dir)

    data = move_to_device(data, device)

    if args.task == "node" and args.dataset in {"A", "B"}:
        model, best_metric = train_node_task(data, args.dataset, device)
    elif args.task == "link" and args.dataset == "C":
        model, best_metric = train_link_task(data, device)
    else:
        raise ValueError("Invalid dataset/task combination.")

    path = save_checkpoint(
        model=model,
        model_dir=args.model_dir,
        kerberos=args.kerberos,
        dataset=args.dataset,
    )

    print(f"Saved model to: {path}")
    print(f"Best validation metric: {best_metric:.6f}")


if __name__ == "__main__":
    main()