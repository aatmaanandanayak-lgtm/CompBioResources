from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Message-passing layer

class MPNNLayer(nn.Module):
    """
    A single message-passing layer with edge-conditioned messages.

    For each node i the update rule is:

        m_ij  = MLP_msg( h_i || h_j || e_ij )
        h_i'  = MLP_upd( h_i || Σ_j m_ij )

    where || denotes concatenation and the sum is over neighbours j of i.

    Parameters
    ----------
    node_dim : int
        Dimensionality of node hidden representations.
    edge_dim : int
        Dimensionality of edge feature vectors.
    hidden_dim : int
        Width of the internal MLPs.
    dropout : float
        Dropout probability applied after each layer normalisation.
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        msg_in = 2 * node_dim + edge_dim
        self.message_mlp = nn.Sequential(
            nn.Linear(msg_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        upd_in = node_dim + hidden_dim
        self.update_mlp = nn.Sequential(
            nn.Linear(upd_in, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, node_dim),
        )
        self.norm = nn.LayerNorm(node_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        h: Tensor,            # (N, node_dim)
        edge_index: Tensor,   # (2, E)  — [src, dst]
        edge_attr: Tensor,    # (E, edge_dim)
    ) -> Tensor:
        src, dst = edge_index[0], edge_index[1]

        # Construct messages: one per directed edge
        msg_input = torch.cat([h[src], h[dst], edge_attr], dim=-1)   # (E, msg_in)
        messages = self.message_mlp(msg_input)                        # (E, hidden_dim)

        # Aggregate messages at destination nodes (mean aggregation)
        N = h.size(0)
        agg = torch.zeros(N, messages.size(-1), device=h.device)
        count = torch.zeros(N, 1, device=h.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(messages), messages)
        count.scatter_add_(0, dst.unsqueeze(-1), torch.ones(dst.size(0), 1, device=h.device))
        count = count.clamp(min=1.0)
        agg = agg / count                                             # (N, hidden_dim)

        # Update node representations with residual connection
        upd_input = torch.cat([h, agg], dim=-1)                      # (N, node_dim + hidden_dim)
        h_new = self.update_mlp(upd_input)                           # (N, node_dim)
        h_new = self.dropout(self.norm(h_new + h))                   # residual
        return h_new



# Full GNN model

class AllostericGNN(nn.Module):
    """
    Multi-layer MPNN with dual output heads for allosteric site prediction
    and pathway edge scoring.

    Parameters
    ----------
    in_node_dim : int
        Dimensionality of raw input node features.
    in_edge_dim : int
        Dimensionality of raw input edge features.
    hidden_dim : int
        Width of all hidden layers.
    num_layers : int
        Number of message-passing iterations.
    dropout : float
        Dropout applied within each MP layer and output heads.
    predict_pathways : bool
        If True, attach the edge-scoring head and return edge scores.
    """

    def __init__(
        self,
        in_node_dim: int = 30,
        in_edge_dim: int = 5,
        hidden_dim: int = 128,
        num_layers: int = 4,
        dropout: float = 0.1,
        predict_pathways: bool = True,
    ) -> None:
        super().__init__()
        self.predict_pathways = predict_pathways

        # Input projections
        self.node_embed = nn.Sequential(
            nn.Linear(in_node_dim, hidden_dim),
            nn.SiLU(),
        )
        self.edge_embed = nn.Sequential(
            nn.Linear(in_edge_dim, hidden_dim),
            nn.SiLU(),
        )

        # Message-passing stack
        self.mp_layers = nn.ModuleList([
            MPNNLayer(
                node_dim=hidden_dim,
                edge_dim=hidden_dim,
                hidden_dim=hidden_dim,
                dropout=dropout,
            )
            for _ in range(num_layers)
        ])

        # Node classification head: P(residue is allosteric)
        self.node_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        # Edge scoring head: allosteric pathway importance w_ij
        if predict_pathways:
            self.edge_scorer = nn.Sequential(
                nn.Linear(2 * hidden_dim + hidden_dim, hidden_dim // 2),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

    def forward(
        self,
        x: Tensor,            # (N, in_node_dim)
        edge_index: Tensor,   # (2, E)
        edge_attr: Tensor,    # (E, in_edge_dim)
    ) -> dict[str, Tensor]:
        """
        Forward pass.

        Returns
        -------
        dict with keys:
            "node_probs"  : Tensor (N,) — P(allosteric) per residue
            "edge_scores" : Tensor (E,) — pathway importance per edge
                            (only present if predict_pathways=True)
        """
        h = self.node_embed(x)               # (N, hidden_dim)
        e = self.edge_embed(edge_attr)        # (E, hidden_dim)

        for layer in self.mp_layers:
            h = layer(h, edge_index, e)

        # Node head
        node_logits = self.node_classifier(h).squeeze(-1)    # (N,)
        node_probs = torch.sigmoid(node_logits)

        out = {"node_probs": node_probs, "node_logits": node_logits}

        # Edge head
        if self.predict_pathways:
            src, dst = edge_index[0], edge_index[1]
            edge_repr = torch.cat([h[src], h[dst], e], dim=-1)   # (E, 3*hidden_dim)
            edge_logits = self.edge_scorer(edge_repr).squeeze(-1) # (E,)
            out["edge_scores"] = torch.sigmoid(edge_logits)
            out["edge_logits"] = edge_logits

        return out


# Focal loss (imbalanced classification)

class FocalLoss(nn.Module):
    """
    Focal loss for binary classification under class imbalance.

    FL(p) = -α · (1 - p)^γ · log(p)

    Parameters
    ----------
    alpha : float
        Weighting factor for the positive (allosteric) class.
    gamma : float
        Focusing parameter. Higher values down-weight easy negatives more.
    """

    def __init__(self, alpha: float = 0.75, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: Tensor, targets: Tensor) -> Tensor:
        """
        Parameters
        ----------
        logits : Tensor (N,) — raw (pre-sigmoid) predictions
        targets : Tensor (N,) — binary labels {0, 1}
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets.float(), reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1.0 - probs)
        alpha_t = torch.where(targets == 1,
                              torch.full_like(pt, self.alpha),
                              torch.full_like(pt, 1.0 - self.alpha))
        focal_weight = alpha_t * (1.0 - pt) ** self.gamma
        return (focal_weight * bce).mean()
