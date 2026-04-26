import os
import torch


def get_model_path(model_dir: str, kerberos: str, dataset: str) -> str:
    return os.path.join(model_dir, f"{kerberos}_model_{dataset}.pt")


def save_checkpoint(model, model_dir: str, kerberos: str, dataset: str) -> str:
    os.makedirs(model_dir, exist_ok=True)
    path = get_model_path(model_dir, kerberos, dataset)

    model = model.to("cpu")
    model.eval()
    torch.save(model, path)

    return path