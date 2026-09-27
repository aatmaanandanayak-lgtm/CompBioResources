from __future__ import annotations

import argparse
import json
import os
import pickle
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from gnn_allostery.models.allosteric_gnn import AllostericGNN, FocalLoss
from gnn_allostery.utils.graph_construction import ProteinGraph

# Dataset helpers

def load_graphs(data_dir: str) -> list[ProteinGraph]:
    """Load all pickled ProteinGraph objects from a directory."""
    graphs = []
    for path in sorted(Path(data_dir).glob("*.pkl")):
        with open(path, "rb") as fh:
            graphs.append(pickle.load(fh))
    if not graphs:
        raise FileNotFoundError(f"No .pkl graph files found in {data_dir}.")
    return graphs


def graph_to_tensors(
    graph: ProteinGraph,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Convert a ProteinGraph to PyTorch tensors on the given device."""
    x = torch.tensor(graph.node_features, dtype=torch.float32, device=device)
    ei = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
    ea = torch.tensor(graph.edge_features, dtype=torch.float32, device=device)
    labels = (
        torch.tensor(graph.labels, dtype=torch.long, device=device)
        if graph.labels is not None else None
    )
    return x, ei, ea, labels

# Metrics

def compute_metrics(
    all_probs: np.ndarray,
    all_labels: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, float]:
    """Compute ROC-AUC, PR-AUC and F1 for the positive (allosteric) class."""
    preds = (all_probs >= threshold).astype(int)
    metrics: dict[str, float] = {}
    try:
        metrics["roc_auc"] = roc_auc_score(all_labels, all_probs)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["pr_auc"] = average_precision_score(all_labels, all_probs)
    except ValueError:
        metrics["pr_auc"] = float("nan")
    metrics["f1"] = f1_score(all_labels, preds, zero_division=0)
    return metrics

# Training and validation steps

def run_epoch(
    model: AllostericGNN,
    graphs: list[ProteinGraph],
    node_loss_fn: FocalLoss,
    edge_loss_fn: nn.BCEWithLogitsLoss,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
    edge_loss_weight: float = 0.5,
    train: bool = True,
) -> tuple[float, dict[str, float]]:
    """
    Run one full pass over a list of protein graphs.

    Returns
    -------
    mean_loss : float
    metrics   : dict with keys roc_auc, pr_auc, f1
    """
    model.train(train)
    total_loss = 0.0
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for graph in graphs:
            x, ei, ea, labels = graph_to_tensors(graph, device)
            if labels is None:
                continue

            out = model(x, ei, ea)
            node_logits = out["node_logits"]

            # Node classification loss
            loss = node_loss_fn(node_logits, labels)

            # Optional edge pathway loss (requires edge labels on graph)
            if model.predict_pathways and hasattr(graph, "edge_labels") and graph.edge_labels is not None:
                edge_labels = torch.tensor(
                    graph.edge_labels, dtype=torch.float32, device=device
                )
                loss = loss + edge_loss_weight * edge_loss_fn(
                    out["edge_logits"], edge_labels
                )

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            total_loss += loss.item()
            all_probs.append(out["node_probs"].detach().cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_probs_arr = np.concatenate(all_probs) if all_probs else np.array([])
    all_labels_arr = np.concatenate(all_labels) if all_labels else np.array([])
    metrics = compute_metrics(all_probs_arr, all_labels_arr) if len(all_labels_arr) > 0 else {}
    return total_loss / max(len(graphs), 1), metrics

# Main training entry point

def train(
    data_dir: str,
    output_dir: str,
    hidden_dim: int = 128,
    num_layers: int = 4,
    dropout: float = 0.1,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    epochs: int = 100,
    patience: int = 15,
    val_fraction: float = 0.15,
    focal_alpha: float = 0.75,
    focal_gamma: float = 2.0,
    edge_loss_weight: float = 0.5,
    predict_pathways: bool = False,
    seed: int = 42,
    device_str: str = "cpu",
) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str)
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading graphs from {data_dir} ...")
    graphs = load_graphs(data_dir)
    print(f"  {len(graphs)} graphs loaded.")

    # Train / validation split (random; sequence-identity clustering
    # should be applied externally to ensure non-redundancy)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(graphs))
    n_val = max(1, int(val_fraction * len(graphs)))
    val_graphs = [graphs[i] for i in indices[:n_val]]
    train_graphs = [graphs[i] for i in indices[n_val:]]
    print(f"  Train: {len(train_graphs)}  Val: {len(val_graphs)}")

    # Infer feature dimensions from first graph
    in_node_dim = graphs[0].node_features.shape[1]
    in_edge_dim = graphs[0].edge_features.shape[1] if graphs[0].edge_features.ndim == 2 else 5

    model = AllostericGNN(
        in_node_dim=in_node_dim,
        in_edge_dim=in_edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        predict_pathways=predict_pathways,
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    node_loss_fn = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    edge_loss_fn = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_auc = -1.0
    epochs_no_improve = 0
    history: list[dict] = []

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_metrics = run_epoch(
            model, train_graphs, node_loss_fn, edge_loss_fn,
            optimizer, device, edge_loss_weight, train=True,
        )
        val_loss, val_metrics = run_epoch(
            model, val_graphs, node_loss_fn, edge_loss_fn,
            None, device, edge_loss_weight, train=False,
        )
        scheduler.step()

        elapsed = time.time() - t0
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)

        val_auc = val_metrics.get("roc_auc", -1.0)
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | "
            f"val_roc_auc={val_auc:.4f} | val_pr_auc={val_metrics.get('pr_auc', float('nan')):.4f} | "
            f"val_f1={val_metrics.get('f1', float('nan')):.4f} | {elapsed:.1f}s"
        )

        # Checkpoint on improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            epochs_no_improve = 0
            ckpt_path = os.path.join(output_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  ✓ New best val ROC-AUC={best_val_auc:.4f} — checkpoint saved.")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {epoch} epochs.")
                break

    # Save training history
    hist_path = os.path.join(output_dir, "training_history.json")
    with open(hist_path, "w") as fh:
        json.dump(history, fh, indent=2)
    print(f"Training complete. Best val ROC-AUC: {best_val_auc:.4f}")
    print(f"Checkpoints and history saved to {output_dir}.")

# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train AllostericGNN.")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--predict_pathways", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        lr=args.lr,
        epochs=args.epochs,
        patience=args.patience,
        predict_pathways=args.predict_pathways,
        device_str=args.device,
    )
