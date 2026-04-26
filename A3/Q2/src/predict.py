"""
predict.py for COL761 Assignment 3.

Usage:
python predict.py --dataset A|B|C --task node|link --data_dir /absolute/path/to/data_dir \
    --model_dir /path/to/models --output_dir /path/to/outputs --kerberos YOUR_KERBEROS
"""

import argparse
import os

import numpy as np
import torch

from load_dataset import COL761NodeDataset, COL761LinkDataset, load_dataset, _load_edge_list


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def move_to_device(obj, device):
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


def load_model(model_path: str, device: torch.device) -> torch.nn.Module:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    model = torch.load(model_path, weights_only=False, map_location=device)
    model = model.to(device)
    model.eval()

    return model


def _random_A(dataset: COL761NodeDataset) -> torch.Tensor:
    return torch.randint(0, dataset.num_classes, (dataset[0].num_nodes,))


def _random_B(dataset: COL761NodeDataset) -> torch.Tensor:
    return torch.rand(dataset[0].num_nodes)


def _random_C(V: int, K: int):
    return torch.rand(V), torch.rand(V, K)


def _flatten_edge_pairs(edge_pairs: torch.Tensor) -> torch.Tensor:
    edge_pairs = edge_pairs.long()

    if edge_pairs.dim() == 3:
        return edge_pairs.view(-1, 2).contiguous()

    if edge_pairs.dim() == 2 and edge_pairs.size(0) == 2:
        return edge_pairs.t().contiguous()

    if edge_pairs.dim() == 2 and edge_pairs.size(1) == 2:
        return edge_pairs.contiguous()

    raise ValueError(f"Invalid edge-pair shape: {tuple(edge_pairs.shape)}")


@torch.no_grad()
def predict_A(model: torch.nn.Module, dataset: COL761NodeDataset, device: torch.device) -> torch.Tensor:
    data = move_to_device(dataset[0], device)

    logits = model(data.x.float(), data.edge_index.long())
    y_pred = logits.argmax(dim=1)

    return y_pred.cpu().long()


@torch.no_grad()
def predict_B(model: torch.nn.Module, dataset: COL761NodeDataset, device: torch.device) -> torch.Tensor:
    data = move_to_device(dataset[0], device)

    logits = model(data.x.float(), data.edge_index.long())

    if logits.dim() == 1:
        y_score = torch.sigmoid(logits)
    elif logits.shape[1] == 1:
        y_score = torch.sigmoid(logits).view(-1)
    else:
        y_score = torch.softmax(logits, dim=1)[:, 1]

    return y_score.cpu().float()


@torch.no_grad()
def decode_edges_in_batches(model, z, edge_pairs, batch_size: int = 262144) -> torch.Tensor:
    edge_pairs = _flatten_edge_pairs(edge_pairs)

    scores = []

    for start in range(0, edge_pairs.size(0), batch_size):
        end = min(start + batch_size, edge_pairs.size(0))
        score = model.decode(z, edge_pairs[start:end])
        scores.append(score.detach().cpu())

    return torch.cat(scores, dim=0)


@torch.no_grad()
def predict_C(
    model: torch.nn.Module,
    dataset: COL761LinkDataset,
    device: torch.device,
    test_dir: str = None,
):
    dataset = move_to_device(dataset, device)

    if test_dir is None:
        pos = dataset.valid_pos
        neg = dataset.valid_neg
        split = "valid"
    else:
        pos = _load_edge_list(os.path.join(test_dir, "test_pos.txt"))

        npy = os.path.join(test_dir, "test_neg_hard.npy")
        with open(npy, "rb") as f:
            neg = torch.from_numpy(np.load(f))

        pos = pos.to(device)
        neg = neg.to(device)
        split = "test"

    pos = _flatten_edge_pairs(pos).to(device)
    neg = neg.long().to(device)

    if neg.dim() != 3:
        raise ValueError("Negative candidates for C must have shape [P, K, 2].")

    P, K, _ = neg.shape

    if hasattr(model, "encode") and hasattr(model, "decode"):
        z = model.encode(dataset.x.float(), dataset.edge_index.long())

        pos_scores = decode_edges_in_batches(
            model=model,
            z=z,
            edge_pairs=pos,
            batch_size=262144,
        )

        neg_scores = decode_edges_in_batches(
            model=model,
            z=z,
            edge_pairs=neg.view(P * K, 2),
            batch_size=262144,
        ).view(P, K)

    else:
        pos_scores = model(dataset.x.float(), dataset.edge_index.long(), pos).detach().cpu()
        neg_scores = model(
            dataset.x.float(),
            dataset.edge_index.long(),
            neg.view(P * K, 2),
        ).view(P, K).detach().cpu()

    return pos_scores.float(), neg_scores.float(), split


def predict_and_save(
    dataset_name: str,
    data_dir: str,
    model_path: str,
    out_dir: str,
    test_dir: str = None,
    kerberos: str = "student",
) -> None:
    os.makedirs(out_dir, exist_ok=True)

    device = get_device()
    print(f"Using device: {device}")
    print(f"Loading dataset {dataset_name} ...")

    ds = load_dataset(dataset_name, data_dir)

    if model_path is not None:
        print(f"Loading model from {model_path} ...")
        model = load_model(model_path, device)
    else:
        print("No --model_dir given, using random predictions.")
        model = None

    if dataset_name == "A":
        y_pred = predict_A(model, ds, device) if model is not None else _random_A(ds)

        assert y_pred.shape == (ds[0].num_nodes,)
        assert y_pred.dtype == torch.long

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_A.pt")
        torch.save({"y_pred": y_pred.cpu()}, out_path)

        print(f"Saved {out_path} shape={tuple(y_pred.shape)}")

    elif dataset_name == "B":
        y_score = predict_B(model, ds, device) if model is not None else _random_B(ds)

        assert y_score.shape == (ds[0].num_nodes,)
        assert y_score.is_floating_point()

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_B.pt")
        torch.save({"y_score": y_score.cpu()}, out_path)

        print(f"Saved {out_path} shape={tuple(y_score.shape)}")

    elif dataset_name == "C":
        if model is not None:
            pos_scores, neg_scores, split = predict_C(model, ds, device, test_dir=test_dir)
        else:
            if test_dir or not hasattr(ds, "valid_pos"):
                pos = ds.test_pos
                neg = ds.test_neg
                split = "test"
            else:
                pos = ds.valid_pos
                neg = ds.valid_neg
                split = "valid"

            V, K = pos.shape[0], neg.shape[1]
            pos_scores, neg_scores = _random_C(V, K)

        out_path = os.path.join(out_dir, f"{kerberos}_predictions_C.pt")
        torch.save(
            {
                "pos_scores": pos_scores.cpu(),
                "neg_scores": neg_scores.cpu(),
                "split": split,
            },
            out_path,
        )

        print(f"Saved {out_path} split={split}")
        print(f"  pos_scores: {tuple(pos_scores.shape)}")
        print(f"  neg_scores: {tuple(neg_scores.shape)}")


def main():
    parser = argparse.ArgumentParser(description="Generate predictions for COL761 Assignment 3.")
    parser.add_argument("--dataset", required=True, choices=["A", "B", "C"])
    parser.add_argument("--task", required=True, choices=["node", "link"])
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--model_dir", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--kerberos", required=True)
    parser.add_argument("--test_dir", default=None, help=argparse.SUPPRESS)

    args = parser.parse_args()

    valid = {"node": ("A", "B"), "link": ("C",)}
    if args.dataset not in valid[args.task]:
        parser.error(
            f"--task {args.task} is not valid for --dataset {args.dataset}. "
            f"Expected dataset in {valid[args.task]}."
        )

    if not os.path.isabs(args.data_dir):
        parser.error("--data_dir must be an absolute path")

    model_path = None
    if args.model_dir is not None:
        model_path = os.path.join(
            args.model_dir,
            f"{args.kerberos}_model_{args.dataset}.pt",
        )

    predict_and_save(
        dataset_name=args.dataset,
        data_dir=args.data_dir,
        model_path=model_path,
        out_dir=args.output_dir,
        test_dir=args.test_dir,
        kerberos=args.kerberos,
    )


if __name__ == "__main__":
    main()