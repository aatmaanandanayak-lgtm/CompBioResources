"""
rouse_polymer/simulation/rouse_chain.py

Rouse polymer model simulations for studying protein folding kinetics under
physiologically relevant constraints.

Background
The Rouse model treats a polymer chain as N beads connected by harmonic springs (ignoring hydrodynamic
interactions). It provides a good enough description of polypeptide dynamics at CG resolution and provides
the necessary contact formation rates between chain segments as output (needed to discriminate b/w models).

The central tenets of the two models are as such:

  Diffusion-Collision Model (DCM)
      Independently stable microdomains (helices, hairpins) form first, then
      diffuse and collide to assemble the native structure. The rate-limiting
      step is the collision of pre-formed structural units.
      Chaperone confinement accelerates folding by reducing the effective
      diffusion space and increasing collision frequency.

  Nucleation-Condensation Model (NCM)
      Structure forms cooperatively around a weakly stable nucleus; secondary
      and tertiary contacts develop simultaneously. Confinement does not
      straightforwardly (linearly) accelerate folding under this model.

Overdamped Langevin (Brownian) dynamics of a Rouse chain with optional confinement potentials mimic 
the GroEL/ES cavity; first-passage contact times (FPCTs) between designated microdomain
segments are the primary observable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# Parameters

@dataclass
class RouseParams:
    """Parameters for a Rouse chain Brownian dynamics simulation."""

    # Chain properties
    n_beads: int = 50                  # number of residues / coarse-grained beads
    b: float = 0.38                    # statistical segment length (nm), Cα–Cα distance

    # Spring constant (k_B T / b²) sets the harmonic bond strength
    k_spring: float = 1.0              # in units of k_B T / nm²

    # Friction coefficient (determines timescale)
    gamma: float = 1.0                 # friction coefficient (k_B T · ps / nm²)

    # Temperature (sets noise amplitude via fluctuation-dissipation)
    kT: float = 1.0                    # thermal energy (k_B T units, set to 1)

    # Time integration
    dt: float = 1e-4                   # time step (ps)
    n_steps: int = 5_000_000          # total integration steps
    n_runs: int = 100                  # independent trajectories for FPCT statistics

    # Confinement (GroEL/ES cavity model)
    confinement: bool = False
    cavity_radius: float = 7.0        # effective cavity radius (nm)
    confinement_strength: float = 10.0  # k_B T / nm² (soft repulsive wall)

    # Crowding (mean-field effective excluded volume)
    crowding: bool = False
    crowding_strength: float = 0.5    # k_B T (adds to effective spring constant)

    # Contact detection
    contact_pairs: list = field(default_factory=lambda: [(0, 25), (0, 49)])
    contact_radius: float = 1.5       # nm — distance threshold for contact

    # Folding model variant
    model: str = "DCM"                # "DCM" or "NCM"

    # Seed
    seed: int = 0

# Force computation

def compute_forces(
    positions: np.ndarray,
    params: RouseParams,
) -> np.ndarray:
    """
    Compute total force on each bead.

    Contributions:
      1. Harmonic spring forces from bonded neighbours.
      2. Confinement repulsion (spherical soft wall, if enabled).
      3. Crowding-induced effective stiffening (mean-field, if enabled).
      4. Model-specific native contact attraction (DCM or NCM).

    Parameters
    ----------
    positions : np.ndarray (N, 3)
    params : RouseParams

    Returns
    -------
    forces : np.ndarray (N, 3)
    """
    N = params.n_beads
    forces = np.zeros((N, 3))

    # 1. Harmonic bond springs
    for i in range(N - 1):
        r_vec = positions[i + 1] - positions[i]
        r = np.linalg.norm(r_vec)
        if r > 0:
            f = params.k_spring * r_vec  # F = -k * (r - 0) in Rouse (rest length = 0)
            forces[i] += f
            forces[i + 1] -= f

    # 2. Confinement: spherical harmonic repulsion
    if params.confinement:
        for i in range(N):
            r = np.linalg.norm(positions[i])
            if r > params.cavity_radius:
                direction = positions[i] / r
                penetration = r - params.cavity_radius
                forces[i] -= params.confinement_strength * penetration * direction

    # 3. Crowding: add effective uniform compression toward centre of mass
    if params.crowding:
        com = positions.mean(axis=0)
        for i in range(N):
            forces[i] -= params.crowding_strength * (positions[i] - com)

    # 4. Model-specific native contact potentials
    if params.model == "DCM":
        # DCM: attractive Go-like interactions between defined microdomains
        # Here simplified as pairwise attractive Gaussian wells between
        # microdomain centre beads
        microdomain_pairs = [
            (N // 4, 3 * N // 4),   # between microdomain 1 and 2 centres
        ]
        for i, j in microdomain_pairs:
            r_vec = positions[j] - positions[i]
            r = np.linalg.norm(r_vec)
            if r > 0:
                # Weak attractive force (models partially stable microdomains)
                f_mag = 0.3 * np.exp(-0.5 * (r / 2.0)**2) / r
                f = f_mag * r_vec
                forces[i] += f
                forces[j] -= f

    elif params.model == "NCM":
        # NCM: diffuse, cooperative nucleus — no strong pairwise microdomain
        # contacts; instead a weak global compaction tendency
        com = positions.mean(axis=0)
        for i in range(N):
            forces[i] -= 0.05 * (positions[i] - com)

    return forces

# Brownian dynamics integrator (overdamped Langevin)

def brownian_dynamics_step(
    positions: np.ndarray,
    forces: np.ndarray,
    params: RouseParams,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    One step of overdamped Langevin dynamics:

        r(t + dt) = r(t) + (dt / γ) · F(r) + √(2 k_B T dt / γ) · η

    where η ~ N(0, I).
    """
    noise_std = np.sqrt(2.0 * params.kT * params.dt / params.gamma)
    noise = rng.standard_normal(positions.shape) * noise_std
    positions_new = positions + (params.dt / params.gamma) * forces + noise
    return positions_new

# First-passage contact time measurement

def run_single_trajectory(
    params: RouseParams,
    rng: np.random.Generator,
) -> dict[tuple[int, int], Optional[float]]:
    """
    Run one Brownian dynamics trajectory and record the first-passage contact
    time (FPCT) for each specified contact pair.

    A contact is registered when the distance between the two beads first
    falls below `params.contact_radius`.

    Returns
    -------
    dict mapping each contact pair to its FPCT (in ps), or None if not reached.
    """
    N = params.n_beads

    # Initialise as a random coil: Gaussian displacements along chain
    positions = np.zeros((N, 3))
    for i in range(1, N):
        positions[i] = positions[i - 1] + rng.standard_normal(3) * params.b

    # Centre at origin
    positions -= positions.mean(axis=0)

    # Track which pairs have been contacted
    remaining_pairs = set(params.contact_pairs)
    fpct: dict[tuple[int, int], Optional[float]] = {p: None for p in params.contact_pairs}

    for step in range(params.n_steps):
        forces = compute_forces(positions, params)
        positions = brownian_dynamics_step(positions, forces, params, rng)

        # Check contact formation for remaining pairs
        if remaining_pairs:
            for pair in list(remaining_pairs):
                i, j = pair
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist <= params.contact_radius:
                    fpct[pair] = step * params.dt
                    remaining_pairs.discard(pair)

        if not remaining_pairs:
            break

    return fpct

# Multi-trajectory FPCT statistics


def compute_fpct_statistics(params: RouseParams) -> dict:
    """
    Run `params.n_runs` independent trajectories and compute mean, median,
    and standard deviation of first-passage contact times for each pair.

    Returns
    -------
    dict mapping each contact pair to summary statistics.
    """
    rng = np.random.default_rng(params.seed)
    all_fpcts: dict[tuple, list[float]] = {p: [] for p in params.contact_pairs}

    print(f"Running {params.n_runs} trajectories (model={params.model}, "
          f"confinement={params.confinement}, crowding={params.crowding}) ...")

    for run in range(params.n_runs):
        fpct = run_single_trajectory(params, rng)
        for pair, t in fpct.items():
            if t is not None:
                all_fpcts[pair].append(t)
        if (run + 1) % 10 == 0:
            print(f"  {run + 1}/{params.n_runs} trajectories complete.")

    stats = {}
    for pair, times in all_fpcts.items():
        if times:
            arr = np.array(times)
            stats[pair] = {
                "mean_fpct": float(arr.mean()),
                "median_fpct": float(np.median(arr)),
                "std_fpct": float(arr.std()),
                "contact_fraction": len(times) / params.n_runs,
                "n_contacts": len(times),
            }
        else:
            stats[pair] = {
                "mean_fpct": float("inf"),
                "median_fpct": float("inf"),
                "std_fpct": float("nan"),
                "contact_fraction": 0.0,
                "n_contacts": 0,
            }
    return stats

# Model comparison: DCM vs NCM under cellular constraints


def compare_models(base_params: RouseParams) -> dict:
    """
    Run simulations for DCM and NCM under four conditions:
      1. Free chain (no confinement, no crowding)
      2. Confinement only (GroEL/ES cavity)
      3. Crowding only
      4. Confinement + crowding

    For each condition and model, compute FPCT statistics and assess
    whether confinement produces a rate acceleration consistent with
    experimental chaperone data.

    Returns
    -------
    Nested dict: model → condition → pair → statistics
    """
    import copy

    conditions = {
        "free": {"confinement": False, "crowding": False},
        "confinement": {"confinement": True, "crowding": False},
        "crowding": {"confinement": False, "crowding": True},
        "confinement_crowding": {"confinement": True, "crowding": True},
    }

    results = {}
    for model in ["DCM", "NCM"]:
        results[model] = {}
        for cond_name, cond_kwargs in conditions.items():
            params = copy.deepcopy(base_params)
            params.model = model
            for k, v in cond_kwargs.items():
                setattr(params, k, v)
            stats = compute_fpct_statistics(params)
            results[model][cond_name] = stats
            print(f"\n[{model} | {cond_name}]")
            for pair, s in stats.items():
                print(f"  Pair {pair}: mean FPCT = {s['mean_fpct']:.2f} ps, "
                      f"contact fraction = {s['contact_fraction']:.2f}")

    # Compute acceleration ratios relative to free chain
    print("\n--- Rate acceleration upon confinement ---")
    for model in ["DCM", "NCM"]:
        for pair in base_params.contact_pairs:
            t_free = results[model]["free"][pair]["mean_fpct"]
            t_conf = results[model]["confinement_crowding"][pair]["mean_fpct"]
            if t_conf > 0 and not np.isinf(t_conf):
                ratio = t_free / t_conf
                print(f"  {model} pair {pair}: acceleration ratio = {ratio:.2f}×")
            else:
                print(f"  {model} pair {pair}: insufficient contacts to compute ratio.")

    return results

# Entry point


if __name__ == "__main__":
    base_params = RouseParams(
        n_beads=40,
        n_steps=2_000_000,
        n_runs=50,
        contact_pairs=[(0, 19), (0, 39), (10, 30)],
        seed=42,
    )
    results = compare_models(base_params)
