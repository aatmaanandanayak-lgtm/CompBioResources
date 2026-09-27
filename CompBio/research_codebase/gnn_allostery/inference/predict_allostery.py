"""
gnn_allostery/inference/predict_allostery.py

Run a trained AllostericGNN on a novel PDB structure to:
  1. Predict a per-residue allosteric probability map.
  2. Cluster high-probability residues into candidate allosteric pockets.
  3. (Optionally) extract high-importance communication pathways between the
     top-ranked allosteric pocket and a user-specified active/orthosteric site.

Usage
-----
    python predict_allostery.py \
        --pdb path/to/structure.pdb \
        --checkpoint path/to/best_model.pt \
        --active_site_residues 42 43 44 \
        --output_dir results/
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from gnn_allostery.models.allosteric_gnn import AllostericGNN
from gnn_allostery.utils.graph_construction import build_protein_graph, ProteinGraph


# Pocket detection (greedy clustering of high-probability residues)

def cluster_allosteric_residues(
    ca_coords: np.ndarray,
    node_probs: np.ndarray,
    prob_threshold: float = 0.4,
    cluster_radius: float = 10.0,
) -> list[dict]:
    """
    Greedily cluster residues with predicted allosteric probability above
    `prob_threshold` into spatially contiguous pockets.

    Parameters
    ----------
    ca_coords : np.ndarray (N, 3)
    node_probs : np.ndarray (N,)
    prob_threshold : float
        Minimum probability for a residue to be considered as a seed.
    cluster_radius : float
        Maximum Cα–Cα distance (Å) to merge residues into the same pocket.

    Returns
    -------
    List of pocket dicts, sorted by descending mean probability, each with:
        residue_indices, mean_prob, centroid, size
    """
    candidate_idx = np.where(node_probs >= prob_threshold)[0]
    if len(candidate_idx) == 0:
        return []

    # Sort candidates by decreasing probability
    order = candidate_idx[np.argsort(-node_probs[candidate_idx])]
    assigned = np.zeros(len(ca_coords), dtype=bool)
    pockets = []

    for seed in order:
        if assigned[seed]:
            continue
        # Find all unassigned candidates within cluster_radius of this seed
        dists = np.linalg.norm(ca_coords[order] - ca_coords[seed], axis=-1)
        nearby = order[dists <= cluster_radius]
        nearby = nearby[~assigned[nearby]]

        assigned[nearby] = True
        probs_in = node_probs[nearby]
        centroid = ca_coords[nearby].mean(axis=0)
        pockets.append({
            "residue_indices": nearby.tolist(),
            "mean_prob": float(probs_in.mean()),
            "max_prob": float(probs_in.max()),
            "centroid": centroid.tolist(),
            "size": int(len(nearby)),
        })

    pockets.sort(key=lambda p: p["mean_prob"], reverse=True)
    return pockets


# Pathway extraction via highest-weight path search

def extract_top_pathways(
    edge_index: np.ndarray,
    edge_scores: np.ndarray,
    source_residues: list[int],
    target_residues: list[int],
    n_residues: int,
    top_k: int = 5,
) -> list[list[int]]:
    """
    Extract the top-k highest-weight paths between a set of source residues
    (allosteric pocket) and target residues (active site) using a greedy
    best-first search over edge importance scores.

    Parameters
    ----------
    edge_index  : np.ndarray (2, E)
    edge_scores : np.ndarray (E,)  — values in [0, 1]
    source_residues : list of residue indices (allosteric pocket)
    target_residues : list of residue indices (active/orthosteric site)
    n_residues  : int
    top_k       : int  — number of candidate paths to return

    Returns
    -------
    List of paths (each path is an ordered list of residue indices).
    """
    import heapq

    # Build adjacency list weighted by edge score
    adj: dict[int, list[tuple[float, int]]] = {i: [] for i in range(n_residues)}
    for k in range(edge_index.shape[1]):
        i, j = int(edge_index[0, k]), int(edge_index[1, k])
        w = float(edge_scores[k])
        adj[i].append((w, j))

    target_set = set(target_residues)
    found_paths: list[list[int]] = []
    visited_paths: set[tuple[int, ...]] = set()

    for start in source_residues:
        # Max-heap: use negative weight for priority queue (heapq is min-heap)
        # State: (neg_cumulative_weight, current_node, path_so_far)
        heap: list[tuple[float, int, list[int]]] = [(0.0, start, [start])]
        while heap and len(found_paths) < top_k * len(source_residues):
            neg_w, node, path = heapq.heappop(heap)
            if node in target_set:
                key = tuple(path)
                if key not in visited_paths:
                    visited_paths.add(key)
                    found_paths.append(path)
                continue
            for edge_w, neighbour in adj.get(node, []):
                if neighbour not in path:
                    new_path = path + [neighbour]
                    heapq.heappush(heap, (neg_w - edge_w, neighbour, new_path))

    # Sort by total path weight (descending)
    def path_weight(p: list[int]) -> float:
        total = 0.0
        for step in range(len(p) - 1):
            i, j = p[step], p[step + 1]
            for k in range(edge_index.shape[1]):
                if edge_index[0, k] == i and edge_index[1, k] == j:
                    total += float(edge_scores[k])
                    break
        return total

    found_paths.sort(key=path_weight, reverse=True)
    return found_paths[:top_k]

# Main inference routine

def predict(
    pdb_file: str,
    checkpoint_path: str,
    output_dir: str,
    active_site_residues: list[int] | None = None,
    chain_id: str = "A",
    distance_cutoff: float = 8.0,
    prob_threshold: float = 0.4,
    cluster_radius: float = 10.0,
    hidden_dim: int = 128,
    num_layers: int = 4,
    device_str: str = "cpu",
) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    device = torch.device(device_str)

    # Build protein graph (no labels — inference mode)
    print(f"Building protein graph from {pdb_file} ...")
    graph = build_protein_graph(
        pdb_file=pdb_file,
        chain_id=chain_id,
        distance_cutoff=distance_cutoff,
    )
    print(f"  {graph.node_features.shape[0]} residues, {graph.edge_index.shape[1]} edges.")

    in_node_dim = graph.node_features.shape[1]
    in_edge_dim = graph.edge_features.shape[1] if graph.edge_features.ndim == 2 else 5

    # Load model
    predict_pathways = active_site_residues is not None
    model = AllostericGNN(
        in_node_dim=in_node_dim,
        in_edge_dim=in_edge_dim,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        predict_pathways=predict_pathways,
    ).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    x = torch.tensor(graph.node_features, dtype=torch.float32, device=device)
    ei = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
    ea = torch.tensor(graph.edge_features, dtype=torch.float32, device=device)

    with torch.no_grad():
        out = model(x, ei, ea)

    node_probs = out["node_probs"].cpu().numpy()

    # Pocket clustering
    pockets = cluster_allosteric_residues(
        graph.ca_coords, node_probs,
        prob_threshold=prob_threshold,
        cluster_radius=cluster_radius,
    )
    print(f"  {len(pockets)} candidate allosteric pockets detected.")

    results: dict = {
        "pdb_file": pdb_file,
        "residue_ids": graph.residue_ids,
        "node_probs": node_probs.tolist(),
        "pockets": pockets,
    }

    # Pathway extraction
    if predict_pathways and pockets and "edge_scores" in out:
        edge_scores = out["edge_scores"].cpu().numpy()
        top_pocket_residues = pockets[0]["residue_indices"]
        print(f"  Extracting allosteric communication pathways ...")
        pathways = extract_top_pathways(
            edge_index=graph.edge_index,
            edge_scores=edge_scores,
            source_residues=top_pocket_residues,
            target_residues=active_site_residues,
            n_residues=graph.node_features.shape[0],
        )
        results["pathways"] = [
            {"path_residue_indices": p,
             "path_residue_ids": [graph.residue_ids[i] for i in p]}
            for p in pathways
        ]
        print(f"  {len(pathways)} pathways extracted.")

    # Save
    out_path = os.path.join(output_dir, "allosteric_predictions.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2)
    print(f"Results saved to {out_path}.")
    return results

# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict allosteric sites and pathways.")
    parser.add_argument("--pdb", required=True, help="Path to PDB file.")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--active_site_residues", nargs="*", type=int, default=None)
    parser.add_argument("--chain", default="A")
    parser.add_argument("--prob_threshold", type=float, default=0.4)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    predict(
        pdb_file=args.pdb,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        active_site_residues=args.active_site_residues,
        chain_id=args.chain,
        prob_threshold=args.prob_threshold,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        device_str=args.device,
    )
