"""
gnn_allostery/utils/graph_construction.py

Constructs residue-level protein contact graphs from PDB structures for use
as input to the allosteric site prediction GNN. Each residue becomes a node;
edges are drawn between residues whose Cα atoms lie within a distance cutoff.
Node and edge features are assembled from structural, physicochemical, and
evolutionary data.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from Bio.PDB import PDBParser, DSSP
from Bio.PDB.ResidueDepth import ResidueDepth, get_surface


# ---------------------------------------------------------------------------
# Amino-acid property lookup tables
# ---------------------------------------------------------------------------

AA_ONE_LETTER = list("ACDEFGHIKLMNPQRSTVWY")
AA_INDEX = {aa: i for i, aa in enumerate(AA_ONE_LETTER)}

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "A": 1.8,  "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8,  "K": -3.9, "M": 1.9,  "F": 2.8,  "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Partial charge proxy (simplified formal charge at pH 7)
CHARGE = {
    "A": 0,  "R": 1,  "N": 0,  "D": -1, "C": 0,
    "Q": 0,  "E": -1, "G": 0,  "H": 0,  "I": 0,
    "L": 0,  "K": 1,  "M": 0,  "F": 0,  "P": 0,
    "S": 0,  "T": 0,  "W": 0,  "Y": 0,  "V": 0,
}

# Van der Waals volume (Å³, approximate)
VDW_VOLUME = {
    "A": 67,  "R": 148, "N": 96,  "D": 91,  "C": 86,
    "Q": 114, "E": 109, "G": 48,  "H": 118, "I": 124,
    "L": 124, "K": 135, "M": 124, "F": 135, "P": 90,
    "S": 73,  "T": 93,  "W": 163, "Y": 141, "V": 105,
}

SS_INDEX = {"H": 0, "E": 1, "C": 2, "-": 2}  # helix, sheet, coil


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class ResidueFeatures:
    """Per-residue feature vector assembled from structural and sequence data."""
    residue_index: int
    aa_onehot: np.ndarray          # shape (20,)
    hydrophobicity: float
    charge: float
    vdw_volume: float
    secondary_structure: np.ndarray  # shape (3,) one-hot
    sasa: float                    # solvent-accessible surface area (Å²)
    bfactor: float                 # crystallographic B-factor
    residue_depth: float           # distance from molecular surface (Å)
    conservation: float            # column-wise conservation score from MSA
    ca_coords: np.ndarray          # shape (3,) Cα xyz coordinates

    def to_vector(self) -> np.ndarray:
        """Concatenate all scalar and vector features into a flat array."""
        scalars = np.array([
            self.hydrophobicity,
            self.charge,
            self.vdw_volume,
            self.sasa,
            self.bfactor,
            self.residue_depth,
            self.conservation,
        ], dtype=np.float32)
        return np.concatenate([
            self.aa_onehot.astype(np.float32),
            self.secondary_structure.astype(np.float32),
            scalars,
        ])  # total dimension: 20 + 3 + 7 = 30


@dataclass
class ProteinGraph:
    """
    Residue-level contact graph for a single protein structure.

    Attributes
    ----------
    node_features : np.ndarray, shape (N, F_node)
        Feature matrix; row i corresponds to residue i.
    edge_index : np.ndarray, shape (2, E)
        COO-format edge list; edge_index[0] = source nodes,
        edge_index[1] = target nodes.
    edge_features : np.ndarray, shape (E, F_edge)
        Feature matrix; row k corresponds to edge k.
    labels : np.ndarray or None, shape (N,)
        Binary allosteric labels (1 = allosteric residue, 0 = other).
        None for unlabelled (inference) inputs.
    residue_ids : list[str]
        Chain + residue-number identifiers, e.g. ["A_10", "A_11", ...].
    ca_coords : np.ndarray, shape (N, 3)
        Cα coordinates, retained for downstream pocket clustering.
    """
    node_features: np.ndarray
    edge_index: np.ndarray
    edge_features: np.ndarray
    labels: Optional[np.ndarray]
    residue_ids: list
    ca_coords: np.ndarray


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------

def _run_dssp(model, pdb_file: str) -> dict:
    """Return DSSP secondary-structure and SASA assignments keyed by residue."""
    dssp = DSSP(model, pdb_file, dssp="mkdssp")
    return {key: dssp[key] for key in dssp.property_keys}


def _compute_residue_depths(model) -> dict:
    """Return mean residue depths (Å) keyed by (chain_id, res_seq_num)."""
    surface = get_surface(model)
    rd = ResidueDepth(model, surface)
    depths = {}
    for chain in model:
        for residue in chain:
            key = (chain.id, residue.get_id()[1])
            try:
                depths[key] = rd[chain.id, residue.get_id()][0]
            except (KeyError, TypeError):
                depths[key] = 0.0
    return depths


def _aa_onehot(resname: str) -> np.ndarray:
    vec = np.zeros(20, dtype=np.float32)
    idx = AA_INDEX.get(resname, None)
    if idx is not None:
        vec[idx] = 1.0
    return vec


def _ss_onehot(ss_code: str) -> np.ndarray:
    vec = np.zeros(3, dtype=np.float32)
    vec[SS_INDEX.get(ss_code, 2)] = 1.0
    return vec


# ---------------------------------------------------------------------------
# Edge feature computation
# ---------------------------------------------------------------------------

def _compute_edge_features(
    coords_i: np.ndarray,
    coords_j: np.ndarray,
    coevo_score: float = 0.0,
    is_hbond: bool = False,
    is_salt_bridge: bool = False,
    same_ss: bool = False,
) -> np.ndarray:
    """
    Assemble edge feature vector between residues i and j.

    Features
    --------
    distance         : Euclidean Cα–Cα distance (Å)
    coevo_score      : direct coupling analysis (DCA) coupling strength
    is_hbond         : binary flag for hydrogen bond
    is_salt_bridge   : binary flag for salt bridge
    same_ss_element  : binary flag — share same secondary structure element
    """
    dist = float(np.linalg.norm(coords_i - coords_j))
    return np.array(
        [dist, coevo_score, float(is_hbond), float(is_salt_bridge), float(same_ss)],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Main graph builder
# ---------------------------------------------------------------------------

def build_protein_graph(
    pdb_file: str,
    conservation_scores: Optional[dict] = None,
    coevo_matrix: Optional[np.ndarray] = None,
    allosteric_residues: Optional[set] = None,
    distance_cutoff: float = 8.0,
    chain_id: str = "A",
) -> ProteinGraph:
    """
    Parse a PDB file and construct a residue-level contact graph.

    Parameters
    ----------
    pdb_file : str
        Path to the PDB structure file.
    conservation_scores : dict, optional
        Mapping from residue sequence number to conservation score [0, 1].
        If None, all conservation scores are set to 0.
    coevo_matrix : np.ndarray, optional
        Square matrix of DCA coupling scores indexed by residue position.
        If None, coevolution edge features are set to 0.
    allosteric_residues : set, optional
        Set of residue sequence numbers annotated as allosteric (for training).
        If None, labels are not generated (inference mode).
    distance_cutoff : float
        Cα–Cα distance threshold (Å) for drawing edges. Default 8 Å.
    chain_id : str
        PDB chain identifier to process. Default "A".

    Returns
    -------
    ProteinGraph
        Fully assembled graph ready for input to the GNN.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_file)
    model = structure[0]
    chain = model[chain_id]

    # Collect standard amino-acid residues only
    residues = [
        r for r in chain.get_residues()
        if r.get_id()[0] == " " and r.get_resname() in _three_to_one
    ]

    if len(residues) == 0:
        raise ValueError(f"No standard residues found in chain {chain_id} of {pdb_file}.")

    # DSSP for secondary structure and SASA
    try:
        dssp_data = _run_dssp(model, pdb_file)
    except Exception:
        dssp_data = {}

    # Residue depth
    try:
        depth_data = _compute_residue_depths(model)
    except Exception:
        depth_data = {}

    # Build per-residue feature objects
    res_features: list[ResidueFeatures] = []
    residue_ids: list[str] = []
    ca_coords_list: list[np.ndarray] = []

    for idx, residue in enumerate(residues):
        resname_3 = residue.get_resname().strip()
        aa = _three_to_one.get(resname_3, "G")
        res_seq = residue.get_id()[1]

        # Cα coordinates
        try:
            ca = residue["CA"].get_vector().get_array()
        except KeyError:
            ca = np.zeros(3)

        # B-factor (mean over heavy atoms)
        bfactors = [a.get_bfactor() for a in residue.get_atoms()]
        bfactor = float(np.mean(bfactors)) if bfactors else 0.0

        # DSSP-derived SS and SASA
        dssp_key = (chain_id, residue.get_id())
        if dssp_key in dssp_data:
            ss_code = dssp_data[dssp_key][2]
            sasa = float(dssp_data[dssp_key][3])
        else:
            ss_code = "-"
            sasa = 0.0

        depth_key = (chain_id, res_seq)
        res_depth = float(depth_data.get(depth_key, 0.0))

        conservation = float(
            conservation_scores.get(res_seq, 0.0)
            if conservation_scores else 0.0
        )

        rf = ResidueFeatures(
            residue_index=idx,
            aa_onehot=_aa_onehot(aa),
            hydrophobicity=HYDROPHOBICITY.get(aa, 0.0),
            charge=CHARGE.get(aa, 0.0),
            vdw_volume=VDW_VOLUME.get(aa, 100.0),
            secondary_structure=_ss_onehot(ss_code),
            sasa=sasa,
            bfactor=bfactor,
            residue_depth=res_depth,
            conservation=conservation,
            ca_coords=ca,
        )
        res_features.append(rf)
        residue_ids.append(f"{chain_id}_{res_seq}")
        ca_coords_list.append(ca)

    n_residues = len(res_features)
    ca_array = np.stack(ca_coords_list)  # (N, 3)

    # Pairwise Cα distance matrix
    diff = ca_array[:, None, :] - ca_array[None, :, :]          # (N, N, 3)
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))           # (N, N)

    # Build edge lists
    src_list, dst_list, edge_feat_list = [], [], []
    for i in range(n_residues):
        for j in range(i + 1, n_residues):
            if dist_matrix[i, j] <= distance_cutoff:
                coevo = (
                    float(coevo_matrix[i, j])
                    if coevo_matrix is not None else 0.0
                )
                same_ss = (
                    np.argmax(res_features[i].secondary_structure)
                    == np.argmax(res_features[j].secondary_structure)
                )
                ef = _compute_edge_features(
                    ca_array[i], ca_array[j],
                    coevo_score=coevo,
                    same_ss=same_ss,
                )
                # Undirected: add both directions
                src_list += [i, j]
                dst_list += [j, i]
                edge_feat_list += [ef, ef]

    node_features = np.stack([rf.to_vector() for rf in res_features])  # (N, 30)
    edge_index = np.array([src_list, dst_list], dtype=np.int64)        # (2, E)
    edge_features = np.stack(edge_feat_list) if edge_feat_list else np.zeros((0, 5), dtype=np.float32)

    # Labels
    labels = None
    if allosteric_residues is not None:
        labels = np.array(
            [1 if residues[i].get_id()[1] in allosteric_residues else 0
             for i in range(n_residues)],
            dtype=np.int64,
        )

    return ProteinGraph(
        node_features=node_features,
        edge_index=edge_index,
        edge_features=edge_features,
        labels=labels,
        residue_ids=residue_ids,
        ca_coords=ca_array,
    )


# ---------------------------------------------------------------------------
# Three-letter to one-letter amino acid code mapping
# ---------------------------------------------------------------------------

_three_to_one = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}
