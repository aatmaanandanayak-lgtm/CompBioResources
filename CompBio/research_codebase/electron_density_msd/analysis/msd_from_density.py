"""
electron_density_msd/analysis/msd_from_density.py

Identification of residues with the greatest conformational flexibility from
electron density maps, quantified via mean-square displacement (MSD) estimates.

Scientific background
---------------------
In crystallographic refinement, the Debye-Waller (temperature) factor B
relates to atomic MSD by:

    B = 8π² · <u²>    →    <u²> = B / (8π²)

where <u²> is the isotropic MSD of the atom about its mean position (Å²).

However, B-factors conflate true thermal/conformational flexibility with
model errors and crystal contact effects. A complementary measure of
flexibility can be extracted directly from the electron density map:

  Method 1 — Real-space density variance (RSDV):
    For each residue, compute the variance of density values in the voxels
    surrounding the atom. Diffuse/broad density (high variance across a
    neighbourhood) indicates flexibility; sharp peaks (low local variance)
    indicate rigidity.

  Method 2 — Real-space correlation coefficient (RSCC) / map-model fit:
    Compute the cross-correlation between the observed density map and a
    calculated density (from atomic model) within a sphere around each
    residue. Low RSCC indicates poor map-model agreement, consistent with
    flexibility or disorder.

  Method 3 — Density centroid displacement (for multi-conformer or NMR
    ensembles):
    Fit a Gaussian to the density around each residue in each model/map,
    locate the centroid, and compute the MSD of centroids across conformers.

  Method 4 — B-factor-derived MSD (from PDB):
    Simple conversion of crystallographic B-factors to isotropic MSD.

All four methods are implemented here and can be combined into a consensus
flexibility score.

Outputs
-------
Per-residue flexibility scores, ranked to identify the most mobile regions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit

from electron_density_msd.io.density_io import Atom, ElectronDensityMap


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

B_TO_MSD = 1.0 / (8.0 * np.pi**2)   # MSD (Å²) = B * B_TO_MSD


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------

@dataclass
class ResidueMSD:
    """
    Conformational flexibility estimates for a single residue.

    Attributes
    ----------
    chain : str
    resseq : int
    resname : str
    bfactor_mean : float
        Mean B-factor of Cα atom (Å²).
    msd_bfactor : float
        Isotropic MSD derived from mean B-factor (Å²).
    density_variance : float
        Variance of density values in a sphere around the Cα atom.
    rscc : float
        Real-space correlation coefficient (map-model fit quality).
        Lower = poorer fit = more flexible/disordered. NaN if not computed.
    centroid_msd : float
        MSD of density centroids across ensemble members.
        NaN if not computed (single-structure mode).
    consensus_score : float
        Normalised consensus flexibility score (0 = rigid, 1 = flexible).
    """
    chain: str
    resseq: int
    resname: str
    bfactor_mean: float
    msd_bfactor: float
    density_variance: float
    rscc: float = float("nan")
    centroid_msd: float = float("nan")
    consensus_score: float = float("nan")

    @property
    def residue_id(self) -> str:
        return f"{self.chain}_{self.resseq}_{self.resname}"


# ---------------------------------------------------------------------------
# Utility: extract density values in a sphere around a point
# ---------------------------------------------------------------------------

def extract_sphere(
    density_map: ElectronDensityMap,
    center_xyz: np.ndarray,
    radius_angstrom: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract density values and their Cartesian coordinates within a sphere
    of given radius centred on `center_xyz`.

    Parameters
    ----------
    density_map : ElectronDensityMap
    center_xyz : np.ndarray (3,)
        Centre of the sphere in Cartesian coordinates (Å).
    radius_angstrom : float
        Sphere radius in Å.

    Returns
    -------
    values : np.ndarray
        Density values of voxels within the sphere.
    coords : np.ndarray (M, 3)
        Cartesian coordinates of those voxels.
    """
    dm = density_map
    vs = dm.voxel_size   # (3,)

    # Bounding box in grid units
    r_grid = np.ceil(radius_angstrom / vs).astype(int) + 1

    # Grid index of the centre
    frac_center = center_xyz / np.array([dm.cell_a, dm.cell_b, dm.cell_c])
    grid_center = frac_center * np.array([dm.nx, dm.ny, dm.nz])
    gc = np.round(grid_center).astype(int)

    ix_range = range(max(gc[0] - r_grid[0], 0), min(gc[0] + r_grid[0] + 1, dm.nx))
    iy_range = range(max(gc[1] - r_grid[1], 0), min(gc[1] + r_grid[1] + 1, dm.ny))
    iz_range = range(max(gc[2] - r_grid[2], 0), min(gc[2] + r_grid[2] + 1, dm.nz))

    values = []
    coords = []
    for ix in ix_range:
        for iy in iy_range:
            for iz in iz_range:
                xyz = dm.grid_to_cartesian(np.array([ix, iy, iz], dtype=float))
                dist = np.linalg.norm(xyz - center_xyz)
                if dist <= radius_angstrom:
                    values.append(dm.data[ix, iy, iz])
                    coords.append(xyz)

    return np.array(values, dtype=np.float64), np.array(coords, dtype=np.float64)


# ---------------------------------------------------------------------------
# Method 1: B-factor-derived MSD
# ---------------------------------------------------------------------------

def compute_bfactor_msd(atoms: list[Atom]) -> dict[tuple[str, int], float]:
    """
    Compute per-residue isotropic MSD from Cα B-factors.

    Returns a dict mapping (chain, resseq) → MSD (Å²).
    """
    residue_bfactors: dict[tuple[str, int], list[float]] = {}
    for atom in atoms:
        if atom.is_calpha:
            key = (atom.chain, atom.resseq)
            residue_bfactors.setdefault(key, []).append(atom.bfactor)

    return {
        key: float(np.mean(bfs)) * B_TO_MSD
        for key, bfs in residue_bfactors.items()
    }


# ---------------------------------------------------------------------------
# Method 2: Local density variance around each residue
# ---------------------------------------------------------------------------

def compute_density_variance(
    density_map: ElectronDensityMap,
    atoms: list[Atom],
    radius: float = 2.5,
    calpha_only: bool = True,
) -> dict[tuple[str, int], float]:
    """
    Compute the variance of electron density values in a sphere around
    each residue's Cα atom.

    High variance within the local sphere suggests a broad, diffuse density
    peak consistent with conformational disorder; low variance suggests a
    sharp, well-resolved peak.

    Returns a dict mapping (chain, resseq) → local density variance.
    """
    result: dict[tuple[str, int], float] = {}
    processed: set[tuple[str, int]] = set()

    for atom in atoms:
        if calpha_only and not atom.is_calpha:
            continue
        key = (atom.chain, atom.resseq)
        if key in processed:
            continue
        processed.add(key)

        values, _ = extract_sphere(density_map, atom.coords, radius)
        if len(values) == 0:
            result[key] = 0.0
        else:
            result[key] = float(np.var(values))

    return result


# ---------------------------------------------------------------------------
# Method 3: Real-space correlation coefficient (RSCC)
# ---------------------------------------------------------------------------

def gaussian_density(coords: np.ndarray, center: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate an isotropic 3D Gaussian at given Cartesian coordinates."""
    r2 = np.sum((coords - center[np.newaxis, :]) ** 2, axis=1)
    return np.exp(-r2 / (2 * sigma**2))


def compute_rscc(
    density_map: ElectronDensityMap,
    atoms: list[Atom],
    radius: float = 3.0,
    sigma: float = 0.8,
    calpha_only: bool = True,
) -> dict[tuple[str, int], float]:
    """
    Compute the real-space correlation coefficient (RSCC) for each residue.

    RSCC = Pearson correlation between observed density values and a
    Gaussian model density centred on the atom position within the sphere.

    A high RSCC (close to 1.0) means the atom fits the density well (rigid).
    A low RSCC indicates the model atom does not explain the density well,
    consistent with flexibility, disorder, or model error.

    Parameters
    ----------
    sigma : float
        Width of the Gaussian model density (Å). Approximately the
        resolution-dependent point spread width.
    """
    from scipy.stats import pearsonr

    result: dict[tuple[str, int], float] = {}
    processed: set[tuple[str, int]] = set()

    for atom in atoms:
        if calpha_only and not atom.is_calpha:
            continue
        key = (atom.chain, atom.resseq)
        if key in processed:
            continue
        processed.add(key)

        obs_values, voxel_coords = extract_sphere(density_map, atom.coords, radius)

        if len(obs_values) < 5:
            result[key] = float("nan")
            continue

        # Model density: Gaussian centred on atom
        model_values = gaussian_density(voxel_coords, atom.coords, sigma)

        if obs_values.std() < 1e-10 or model_values.std() < 1e-10:
            result[key] = float("nan")
            continue

        try:
            corr, _ = pearsonr(obs_values, model_values)
            result[key] = float(corr)
        except Exception:
            result[key] = float("nan")

    return result


# ---------------------------------------------------------------------------
# Method 4: Centroid MSD across ensemble (multi-model)
# ---------------------------------------------------------------------------

def compute_centroid_msd(
    density_maps: list[ElectronDensityMap],
    reference_atoms: list[Atom],
    radius: float = 2.5,
    calpha_only: bool = True,
) -> dict[tuple[str, int], float]:
    """
    Compute the MSD of density centroids across an ensemble of maps.

    For each residue, the density centroid is located in each map by
    computing the intensity-weighted mean position of voxels within a sphere
    around the reference atom. The MSD of these centroids across maps gives
    a direct measure of positional uncertainty.

    Parameters
    ----------
    density_maps : list of ElectronDensityMap
        One map per ensemble member (e.g. from time-resolved crystallography
        or MD simulation density).
    reference_atoms : list of Atom
        Atomic positions used as starting points for centroid search.
    """
    if not density_maps:
        return {}

    result: dict[tuple[str, int], float] = {}
    processed: set[tuple[str, int]] = set()

    for atom in reference_atoms:
        if calpha_only and not atom.is_calpha:
            continue
        key = (atom.chain, atom.resseq)
        if key in processed:
            continue
        processed.add(key)

        centroids = []
        for dm in density_maps:
            values, coords = extract_sphere(dm, atom.coords, radius)
            if len(values) < 3 or values.sum() <= 0:
                continue
            # Intensity-weighted centroid
            weights = np.maximum(values, 0.0)
            total_weight = weights.sum()
            if total_weight <= 0:
                continue
            centroid = (coords * weights[:, np.newaxis]).sum(axis=0) / total_weight
            centroids.append(centroid)

        if len(centroids) < 2:
            result[key] = float("nan")
            continue

        centroids_arr = np.stack(centroids)   # (N_maps, 3)
        mean_centroid = centroids_arr.mean(axis=0)
        msd = float(np.mean(np.sum((centroids_arr - mean_centroid) ** 2, axis=1)))
        result[key] = msd

    return result


# ---------------------------------------------------------------------------
# Consensus flexibility scorer
# ---------------------------------------------------------------------------

class ResidueFlexibilityAnalyser:
    """
    Compute and combine multiple flexibility metrics for all residues in a
    structure to produce a ranked list of the most conformationally mobile
    residues.

    Parameters
    ----------
    density_map : ElectronDensityMap
        The primary (e.g. 2mFo-DFc) electron density map.
    atoms : list of Atom
        Atomic model (Cα atoms are used by default).
    ensemble_maps : list of ElectronDensityMap, optional
        Additional maps for centroid MSD calculation.
    sphere_radius : float
        Sphere radius (Å) for local density extraction.
    gaussian_sigma : float
        Gaussian width (Å) for RSCC calculation.
    use_bfactor : bool
        Whether to include B-factor MSD in the consensus score.
    use_density_variance : bool
        Whether to include local density variance.
    use_rscc : bool
        Whether to include RSCC.
    use_centroid_msd : bool
        Whether to include centroid MSD (requires ensemble_maps).
    """

    def __init__(
        self,
        density_map: ElectronDensityMap,
        atoms: list[Atom],
        ensemble_maps: Optional[list[ElectronDensityMap]] = None,
        sphere_radius: float = 2.5,
        gaussian_sigma: float = 0.8,
        use_bfactor: bool = True,
        use_density_variance: bool = True,
        use_rscc: bool = True,
        use_centroid_msd: bool = False,
    ) -> None:
        self.density_map = density_map
        self.atoms = atoms
        self.ensemble_maps = ensemble_maps or []
        self.sphere_radius = sphere_radius
        self.gaussian_sigma = gaussian_sigma
        self.use_bfactor = use_bfactor
        self.use_density_variance = use_density_variance
        self.use_rscc = use_rscc
        self.use_centroid_msd = use_centroid_msd and len(self.ensemble_maps) > 1

    def analyse(self) -> list[ResidueMSD]:
        """
        Run all enabled flexibility metrics and return a ranked list of
        ResidueMSD objects.

        Returns
        -------
        List of ResidueMSD, sorted by consensus_score (descending).
        """
        print("Computing residue flexibility metrics ...")

        # Collect residue metadata
        residue_meta: dict[tuple[str, int], tuple[str, str]] = {}
        for atom in self.atoms:
            if atom.is_calpha:
                key = (atom.chain, atom.resseq)
                residue_meta[key] = (atom.resname, atom.chain)

        all_keys = set(residue_meta.keys())

        # Method 1: B-factor MSD
        bfactor_msd: dict[tuple[str, int], float] = {}
        if self.use_bfactor:
            print("  Computing B-factor MSD ...")
            bfactor_msd = compute_bfactor_msd(self.atoms)

        # Method 2: Density variance
        density_var: dict[tuple[str, int], float] = {}
        if self.use_density_variance:
            print("  Computing local density variance ...")
            density_var = compute_density_variance(
                self.density_map, self.atoms, radius=self.sphere_radius
            )

        # Method 3: RSCC
        rscc_scores: dict[tuple[str, int], float] = {}
        if self.use_rscc:
            print("  Computing RSCC ...")
            rscc_scores = compute_rscc(
                self.density_map, self.atoms,
                radius=self.sphere_radius, sigma=self.gaussian_sigma,
            )

        # Method 4: Centroid MSD
        centroid_msd: dict[tuple[str, int], float] = {}
        if self.use_centroid_msd:
            print(f"  Computing centroid MSD across {len(self.ensemble_maps)} maps ...")
            centroid_msd = compute_centroid_msd(
                self.ensemble_maps, self.atoms, radius=self.sphere_radius
            )

        # Assemble raw scores
        raw_results: list[ResidueMSD] = []
        for key in sorted(all_keys):
            resname, chain = residue_meta[key]
            bfac = bfactor_msd.get(key, float("nan"))
            msd_b = bfac if not np.isnan(bfac) else 0.0
            raw_results.append(ResidueMSD(
                chain=chain,
                resseq=key[1],
                resname=resname,
                bfactor_mean=msd_b / B_TO_MSD if msd_b else float("nan"),
                msd_bfactor=msd_b,
                density_variance=density_var.get(key, float("nan")),
                rscc=rscc_scores.get(key, float("nan")),
                centroid_msd=centroid_msd.get(key, float("nan")),
            ))

        # Normalise each metric to [0, 1] and compute consensus score
        raw_results = self._compute_consensus(raw_results)
        raw_results.sort(key=lambda r: r.consensus_score, reverse=True)

        print(f"  Analysis complete. {len(raw_results)} residues scored.")
        return raw_results

    def _compute_consensus(self, results: list[ResidueMSD]) -> list[ResidueMSD]:
        """
        Normalise each metric and compute a weighted consensus flexibility score.

        Metric directions (high value = flexible):
          - msd_bfactor : high → flexible  (weight +1)
          - density_variance : high → flexible (weight +1)
          - rscc : low → flexible  (weight -1, i.e. invert before normalising)
          - centroid_msd : high → flexible (weight +1)
        """
        def _normalise(vals: np.ndarray) -> np.ndarray:
            finite = vals[np.isfinite(vals)]
            if len(finite) == 0 or finite.std() == 0:
                return np.zeros_like(vals)
            mn, mx = finite.min(), finite.max()
            if mx == mn:
                return np.zeros_like(vals)
            normed = (vals - mn) / (mx - mn)
            normed = np.clip(normed, 0.0, 1.0)
            normed[~np.isfinite(vals)] = 0.5   # impute missing with neutral
            return normed

        n = len(results)
        weights = []
        score_contributions = np.zeros(n)

        if self.use_bfactor:
            bfac_vals = np.array([r.msd_bfactor for r in results])
            norm = _normalise(bfac_vals)
            score_contributions += norm
            weights.append(1.0)

        if self.use_density_variance:
            dv_vals = np.array([r.density_variance for r in results])
            norm = _normalise(dv_vals)
            score_contributions += norm
            weights.append(1.0)

        if self.use_rscc:
            rscc_vals = np.array([r.rscc for r in results])
            norm = _normalise(1.0 - rscc_vals)   # invert: low RSCC = flexible
            score_contributions += norm
            weights.append(1.0)

        if self.use_centroid_msd:
            cmsd_vals = np.array([r.centroid_msd for r in results])
            norm = _normalise(cmsd_vals)
            score_contributions += norm
            weights.append(1.0)

        total_weight = max(sum(weights), 1.0)
        consensus = score_contributions / total_weight

        for i, r in enumerate(results):
            r.consensus_score = float(consensus[i])

        return results


# ---------------------------------------------------------------------------
# Reporting utilities
# ---------------------------------------------------------------------------

def print_flexibility_report(
    results: list[ResidueMSD],
    top_n: int = 20,
) -> None:
    """Print a formatted table of the most flexible residues."""
    header = (
        f"{'Rank':>4}  {'Chain':>5}  {'ResSeq':>6}  {'Res':>4}  "
        f"{'MSD_B(Å²)':>9}  {'Var_ρ':>8}  {'RSCC':>6}  "
        f"{'CMSD(Å²)':>9}  {'Score':>6}"
    )
    print("\n" + "=" * len(header))
    print("Residue Conformational Flexibility Ranking")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for rank, r in enumerate(results[:top_n], start=1):
        rscc_str = f"{r.rscc:.3f}" if np.isfinite(r.rscc) else "  N/A "
        cmsd_str = f"{r.centroid_msd:.4f}" if np.isfinite(r.centroid_msd) else "   N/A   "
        msd_b_str = f"{r.msd_bfactor:.4f}" if np.isfinite(r.msd_bfactor) else "   N/A   "
        dv_str = f"{r.density_variance:.4f}" if np.isfinite(r.density_variance) else "  N/A  "
        print(
            f"{rank:>4}  {r.chain:>5}  {r.resseq:>6}  {r.resname:>4}  "
            f"{msd_b_str:>9}  {dv_str:>8}  {rscc_str:>6}  "
            f"{cmsd_str:>9}  {r.consensus_score:>6.3f}"
        )
    print("=" * len(header))


def save_flexibility_csv(results: list[ResidueMSD], output_path: str) -> None:
    """Save per-residue flexibility scores to a CSV file."""
    import csv
    with open(output_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow([
            "chain", "resseq", "resname",
            "bfactor_mean", "msd_bfactor",
            "density_variance", "rscc",
            "centroid_msd", "consensus_score",
        ])
        for r in results:
            writer.writerow([
                r.chain, r.resseq, r.resname,
                f"{r.bfactor_mean:.4f}",
                f"{r.msd_bfactor:.6f}",
                f"{r.density_variance:.6f}",
                f"{r.rscc:.4f}" if np.isfinite(r.rscc) else "",
                f"{r.centroid_msd:.6f}" if np.isfinite(r.centroid_msd) else "",
                f"{r.consensus_score:.4f}",
            ])
    print(f"Flexibility scores saved to {output_path}.")
