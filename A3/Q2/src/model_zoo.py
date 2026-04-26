import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv


def to_edge_index(edge_pairs: torch.Tensor) -> torch.Tensor:
    """
    Accept either [2, E] or [E, 2] edge representation and return [2, E].
    """
    if edge_pairs.dim() != 2:
        raise ValueError(f"Expected 2D edge tensor, got shape {tuple(edge_pairs.shape)}")

    if edge_pairs.size(0) == 2:
        return edge_pairs.long().contiguous()

    if edge_pairs.size(1) == 2:
        return edge_pairs.t().long().contiguous()

    raise ValueError(f"Cannot interpret edge tensor with shape {tuple(edge_pairs.shape)}")


class NodeGraphSAGE(nn.Module):
    """
    GraphSAGE model for node classification on datasets A and B.
    Dataset A uses multiclass logits.
    Dataset B uses one binary logit per node.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int = 3,
        dropout: float = 0.5,
        input_norm: bool = True,
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        self.dropout = float(dropout)
        self.input_norm = bool(input_norm)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        self.head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, out_channels),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.input_norm:
            x = F.normalize(x.float(), p=2, dim=-1)
        else:
            x = x.float()

        edge_index = to_edge_index(edge_index).to(x.device)

        h = x
        for layer_id, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            if layer_id > 0 and h_new.shape == h.shape:
                h = h + h_new
            else:
                h = h_new

        return h

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.head(z)


class LinkGraphSAGE(nn.Module):
    """
    GraphSAGE encoder with an MLP edge decoder for dataset C link prediction.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_layers: int = 2,
        dropout: float = 0.25,
        input_norm: bool = True,
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers must be at least 2")

        self.dropout = float(dropout)
        self.input_norm = bool(input_norm)

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        self.convs.append(SAGEConv(in_channels, hidden_channels))
        self.norms.append(nn.BatchNorm1d(hidden_channels))

        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.norms.append(nn.BatchNorm1d(hidden_channels))

        edge_feature_dim = 4 * hidden_channels + 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_feature_dim, 2 * hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels, 1),
        )

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.input_norm:
            x = F.normalize(x.float(), p=2, dim=-1)
        else:
            x = x.float()

        edge_index = to_edge_index(edge_index).to(x.device)

        h = x
        for layer_id, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            h_new = conv(h, edge_index)
            h_new = norm(h_new)
            h_new = F.relu(h_new)
            h_new = F.dropout(h_new, p=self.dropout, training=self.training)

            if layer_id > 0 and h_new.shape == h.shape:
                h = h + h_new
            else:
                h = h_new

        return F.normalize(h, p=2, dim=-1)

    def decode(self, z: torch.Tensor, edge_pairs: torch.Tensor) -> torch.Tensor:
        edge_index = to_edge_index(edge_pairs).to(z.device)
        src, dst = edge_index

        z_src = z[src]
        z_dst = z[dst]

        dot = (z_src * z_dst).sum(dim=-1, keepdim=True)

        pair_features = torch.cat(
            [
                z_src,
                z_dst,
                z_src * z_dst,
                torch.abs(z_src - z_dst),
                dot,
            ],
            dim=-1,
        )

        return self.edge_mlp(pair_features).view(-1)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        candidate_edge_index: torch.Tensor,
    ) -> torch.Tensor:
        z = self.encode(x, edge_index)
        return self.decode(z, candidate_edge_index)