"""
mitochondrial_etc/pde_sim/etc_simulation.py

Stochastic spatial simulation of the mitochondrial electron transport chain (ETC)
on a 2D membrane grid using the reaction-diffusion master equation (RDME).

The inner mitochondrial membrane is represented as a 2D lattice. Embedded
respiratory complexes (I, II, III, IV) occupy fixed lattice sites. Mobile
redox carriers — ubiquinol/ubiquinone (QH2/Q) and reduced/oxidised
cytochrome c (CytC_red / CytC_ox) — diffuse stochastically and react when
they encounter the appropriate complex.

Each complex was modelled as a state machine with a simplified set of partial
reactions that preserve physiological stoichiometry and first-order/second-order
kinetics while remaining tractable for lattice-based simulation.

State machines
--------------
Complex I  : IDLE → REDUCED (accepts 2e⁻ from NADH) → IDLE (passes e⁻ to Q)
Complex II : IDLE → REDUCED (accepts 2e⁻ from FADH2) → IDLE (passes e⁻ to Q)
Complex III: IDLE → QH2_BOUND → INTERMEDIATE → IDLE (passes e⁻ to CytC)
Complex IV : IDLE → CYTC_BOUND → IDLE (passes e⁻ to O2, producing H2O)

Outputs
-------
- Time-course of QH2 and CytC_red counts across the membrane.
- Bi-exponential fits of carrier flux time courses.
- Parameter sweep logs for sensitivity analysis.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import curve_fit


# ---------------------------------------------------------------------------
# Physical constants and default parameters
# ---------------------------------------------------------------------------

@dataclass
class SimulationParams:
    """All tuneable parameters for the ETC spatial stochastic simulation."""

    # Grid dimensions
    grid_rows: int = 50
    grid_cols: int = 50
    voxel_size_nm: float = 5.0        # physical size of each lattice voxel (nm)

    # Diffusion coefficients (µm²/s) → converted to jump rates inside sim
    D_ubiquinone: float = 0.5          # lateral diffusion of Q/QH2 in membrane
    D_cytc: float = 5.0               # lateral diffusion of CytC in aqueous phase

    # Initial molecule counts
    n_Q_initial: int = 200            # oxidised ubiquinone molecules at t=0
    n_QH2_initial: int = 50           # reduced ubiquinol molecules at t=0
    n_CytC_ox_initial: int = 100      # oxidised cytochrome c
    n_CytC_red_initial: int = 20      # reduced cytochrome c

    # Complex counts (randomly placed on grid)
    n_complex_I: int = 20
    n_complex_II: int = 10
    n_complex_III: int = 30
    n_complex_IV: int = 20

    # Rate constants (s⁻¹ or s⁻¹ molecule⁻¹ as appropriate)
    k_CI_reduce: float = 50.0         # Complex I: NADH → Complex I reduction rate
    k_CI_Qtransfer: float = 100.0     # Complex I: Q → QH2 transfer rate
    k_CII_reduce: float = 30.0        # Complex II: FADH2 → Complex II reduction rate
    k_CII_Qtransfer: float = 80.0     # Complex II: Q → QH2 transfer rate
    k_CIII_bind: float = 200.0        # Complex III: QH2 binding rate
    k_CIII_transfer: float = 150.0    # Complex III: e⁻ transfer to CytC rate
    k_CIV_bind: float = 300.0         # Complex IV: CytC_red binding rate
    k_CIV_transfer: float = 200.0     # Complex IV: e⁻ transfer to O2 rate

    # Simulation time
    t_end: float = 0.5               # seconds
    dt_record: float = 1e-3          # recording interval (s)

    # Random seed
    seed: int = 0

# Complex state machines

class ComplexState:
    """Generic two-state (IDLE / ACTIVE) model for a respiratory complex."""

    IDLE = 0
    ACTIVE = 1

    def __init__(self, complex_type: str, row: int, col: int) -> None:
        self.complex_type = complex_type
        self.row = row
        self.col = col
        self.state = self.IDLE

    def __repr__(self) -> str:
        return f"{self.complex_type}(r={self.row},c={self.col},state={'IDLE' if self.state == self.IDLE else 'ACTIVE'})"

# RDME-based simulator

class ETCSimulator:
    """
    2D reaction-diffusion master equation simulator of the mitochondrial ETC.

    The membrane is a `grid_rows × grid_cols` lattice. Each voxel can hold
    multiple molecules. At each time step (Next Subvolume Method / Gillespie-
    style event selection) one of the following events is chosen proportionally
    to its propensity:

      - Diffusion hop of a mobile carrier (Q, QH2, CytC_ox, CytC_red) to a
        neighbouring voxel.
      - Reaction between a carrier and a complex occupying the same voxel.
      - Spontaneous complex state transition (e.g. reduction by NADH/FADH2).

    Parameters
    ----------
    params : SimulationParams
    """

    # Species indices
    Q = 0         # ubiquinone (oxidised)
    QH2 = 1       # ubiquinol (reduced)
    CYTC_OX = 2   # cytochrome c (oxidised)
    CYTC_RED = 3  # cytochrome c (reduced)
    N_SPECIES = 4

    def __init__(self, params: SimulationParams) -> None:
        self.p = params
        self.rng = np.random.default_rng(params.seed)

        R, C = params.grid_rows, params.grid_cols
        self.R = R
        self.C = C

        # Molecule count grid: shape (R, C, N_SPECIES)
        self.grid = np.zeros((R, C, self.N_SPECIES), dtype=np.int32)

        # Derived diffusion jump rates (s⁻¹ per voxel hop)
        dx = params.voxel_size_nm * 1e-3   # convert nm → µm
        self.hop_rate_Q = params.D_ubiquinone / dx**2
        self.hop_rate_CytC = params.D_cytc / dx**2

        self.hop_rates = np.array([
            self.hop_rate_Q,    # Q
            self.hop_rate_Q,    # QH2 (same diffusion coefficient)
            self.hop_rate_CytC, # CytC_ox
            self.hop_rate_CytC, # CytC_red
        ])

        # Neighbour offsets (4-connected lattice)
        self.neighbours = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Place complexes and initialise their states
        self.complexes: list[ComplexState] = self._place_complexes()
        self.complex_grid: dict[tuple[int, int], list[ComplexState]] = {}
        for cx in self.complexes:
            key = (cx.row, cx.col)
            self.complex_grid.setdefault(key, []).append(cx)

        # Place mobile carriers uniformly at random
        self._place_carriers()

        # Recording
        self.t_history: list[float] = []
        self.counts_history: list[np.ndarray] = []   # total counts per species


    # Initialisation helpers


    def _place_complexes(self) -> list[ComplexState]:
        """Randomly place respiratory complexes on the grid."""
        positions = self.rng.choice(
            self.R * self.C,
            size=(
                self.p.n_complex_I + self.p.n_complex_II
                + self.p.n_complex_III + self.p.n_complex_IV
            ),
            replace=True,
        )
        flat_to_rc = lambda idx: (idx // self.C, idx % self.C)
        cxs = []
        cursor = 0
        for n, ctype in [
            (self.p.n_complex_I, "CI"),
            (self.p.n_complex_II, "CII"),
            (self.p.n_complex_III, "CIII"),
            (self.p.n_complex_IV, "CIV"),
        ]:
            for k in range(cursor, cursor + n):
                r, c = flat_to_rc(positions[k])
                cxs.append(ComplexState(ctype, r, c))
            cursor += n
        return cxs

    def _place_carriers(self) -> None:
        """Distribute mobile carriers uniformly across the lattice."""
        for species, count in [
            (self.Q, self.p.n_Q_initial),
            (self.QH2, self.p.n_QH2_initial),
            (self.CYTC_OX, self.p.n_CytC_ox_initial),
            (self.CYTC_RED, self.p.n_CytC_red_initial),
        ]:
            rows = self.rng.integers(0, self.R, size=count)
            cols = self.rng.integers(0, self.C, size=count)
            for r, c in zip(rows, cols):
                self.grid[r, c, species] += 1


    # Propensity computation

    def _diffusion_propensities(self) -> tuple[np.ndarray, list]:
        """
        Compute propensities for all possible diffusion events.

        Returns arrays of propensities and corresponding event descriptors.
        """
        props = []
        events = []
        for species in range(self.N_SPECIES):
            rate = self.hop_rates[species]
            counts = self.grid[:, :, species]
            nonzero = np.argwhere(counts > 0)
            for r, c in nonzero:
                n = int(counts[r, c])
                for dr, dc in self.neighbours:
                    nr, nc = (r + dr) % self.R, (c + dc) % self.C
                    props.append(n * rate)
                    events.append(("hop", species, r, c, nr, nc))
        return np.array(props, dtype=np.float64), events

    def _reaction_propensities(self) -> tuple[np.ndarray, list]:
        """Compute propensities for all complex reaction events."""
        props = []
        events = []
        p = self.p
        for cx in self.complexes:
            r, c = cx.row, cx.col
            n_Q = int(self.grid[r, c, self.Q])
            n_QH2 = int(self.grid[r, c, self.QH2])
            n_Cox = int(self.grid[r, c, self.CYTC_OX])
            n_Cred = int(self.grid[r, c, self.CYTC_RED])

            if cx.complex_type == "CI":
                if cx.state == ComplexState.IDLE:
                    props.append(p.k_CI_reduce)
                    events.append(("CI_reduce", cx))
                else:  # ACTIVE: transfer e⁻ to Q
                    prop = p.k_CI_Qtransfer * n_Q
                    props.append(prop)
                    events.append(("CI_Q_transfer", cx))

            elif cx.complex_type == "CII":
                if cx.state == ComplexState.IDLE:
                    props.append(p.k_CII_reduce)
                    events.append(("CII_reduce", cx))
                else:
                    prop = p.k_CII_Qtransfer * n_Q
                    props.append(prop)
                    events.append(("CII_Q_transfer", cx))

            elif cx.complex_type == "CIII":
                if cx.state == ComplexState.IDLE:
                    prop = p.k_CIII_bind * n_QH2
                    props.append(prop)
                    events.append(("CIII_QH2_bind", cx))
                else:
                    prop = p.k_CIII_transfer * n_Cox
                    props.append(prop)
                    events.append(("CIII_CytC_transfer", cx))

            elif cx.complex_type == "CIV":
                if cx.state == ComplexState.IDLE:
                    prop = p.k_CIV_bind * n_Cred
                    props.append(prop)
                    events.append(("CIV_CytC_bind", cx))
                else:
                    props.append(p.k_CIV_transfer)
                    events.append(("CIV_O2_transfer", cx))

        return np.array(props, dtype=np.float64), events


    # Event execution


    def _execute_event(self, event: tuple) -> None:
        """Apply a selected event to the simulation state."""
        etype = event[0]

        if etype == "hop":
            _, species, r, c, nr, nc = event
            if self.grid[r, c, species] > 0:
                self.grid[r, c, species] -= 1
                self.grid[nr, nc, species] += 1

        elif etype == "CI_reduce":
            cx = event[1]
            cx.state = ComplexState.ACTIVE

        elif etype == "CI_Q_transfer":
            cx = event[1]
            r, c = cx.row, cx.col
            if self.grid[r, c, self.Q] > 0:
                self.grid[r, c, self.Q] -= 1
                self.grid[r, c, self.QH2] += 1
                cx.state = ComplexState.IDLE

        elif etype == "CII_reduce":
            cx = event[1]
            cx.state = ComplexState.ACTIVE

        elif etype == "CII_Q_transfer":
            cx = event[1]
            r, c = cx.row, cx.col
            if self.grid[r, c, self.Q] > 0:
                self.grid[r, c, self.Q] -= 1
                self.grid[r, c, self.QH2] += 1
                cx.state = ComplexState.IDLE

        elif etype == "CIII_QH2_bind":
            cx = event[1]
            r, c = cx.row, cx.col
            if self.grid[r, c, self.QH2] > 0:
                self.grid[r, c, self.QH2] -= 1
                self.grid[r, c, self.Q] += 1   # QH2 → Q (e⁻ transferred to CIII)
                cx.state = ComplexState.ACTIVE

        elif etype == "CIII_CytC_transfer":
            cx = event[1]
            r, c = cx.row, cx.col
            if self.grid[r, c, self.CYTC_OX] > 0:
                self.grid[r, c, self.CYTC_OX] -= 1
                self.grid[r, c, self.CYTC_RED] += 1
                cx.state = ComplexState.IDLE

        elif etype == "CIV_CytC_bind":
            cx = event[1]
            r, c = cx.row, cx.col
            if self.grid[r, c, self.CYTC_RED] > 0:
                self.grid[r, c, self.CYTC_RED] -= 1
                self.grid[r, c, self.CYTC_OX] += 1  # CytC_red → CytC_ox
                cx.state = ComplexState.ACTIVE

        elif etype == "CIV_O2_transfer":
            cx = event[1]
            # Electrons transferred to O2 (terminal sink — H2O produced)
            cx.state = ComplexState.IDLE


    # Main simulation loop (Gillespie / Next Subvolume Method)


    def run(self) -> dict:
        """
        Advance the simulation to t_end using the Gillespie SSA.

        Returns
        -------
        dict with keys:
            t          : np.ndarray  — recorded time points
            QH2_counts : np.ndarray  — total QH2 molecules at each time
            CytC_red_counts : np.ndarray
        """
        t = 0.0
        next_record = 0.0
        self._record(t)

        print(f"Starting ETC simulation (t_end={self.p.t_end} s) ...")
        wall_start = time.time()

        while t < self.p.t_end:
            # Compute all propensities
            diff_props, diff_events = self._diffusion_propensities()
            rxn_props, rxn_events = self._reaction_propensities()

            all_props = np.concatenate([diff_props, rxn_props])
            all_events = diff_events + rxn_events

            total_prop = all_props.sum()
            if total_prop == 0.0:
                print("All propensities zero — simulation halted early.")
                break

            # Sample time to next event (exponential)
            dt = self.rng.exponential(1.0 / total_prop)
            t += dt

            # Select event
            cumprobs = np.cumsum(all_props) / total_prop
            u = self.rng.uniform()
            idx = int(np.searchsorted(cumprobs, u))
            idx = min(idx, len(all_events) - 1)
            self._execute_event(all_events[idx])

            # Record at fixed intervals
            if t >= next_record:
                self._record(t)
                next_record += self.p.dt_record

        elapsed = time.time() - wall_start
        print(f"Simulation complete in {elapsed:.2f} s wall time. {len(self.t_history)} time points recorded.")

        t_arr = np.array(self.t_history)
        counts_arr = np.array(self.counts_history)
        return {
            "t": t_arr,
            "QH2_counts": counts_arr[:, self.QH2],
            "CytC_red_counts": counts_arr[:, self.CYTC_RED],
            "Q_counts": counts_arr[:, self.Q],
            "CytC_ox_counts": counts_arr[:, self.CYTC_OX],
        }

    def _record(self, t: float) -> None:
        self.t_history.append(t)
        self.counts_history.append(self.grid.sum(axis=(0, 1)).copy())


# Bi-exponential fitting of flux time courses


def biexponential(t: np.ndarray, A1: float, k1: float, A2: float, k2: float, C: float) -> np.ndarray:
    """f(t) = A1·exp(-k1·t) + A2·exp(-k2·t) + C"""
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + C


def fit_biexponential(
    t: np.ndarray,
    signal: np.ndarray,
) -> dict:
    """
    Fit a bi-exponential decay to a carrier count time course.

    Returns
    -------
    dict with fitted parameters A1, k1, A2, k2, C and the fitted curve.
    """
    # Initial guesses: dominant fast and slow components
    A0 = float(signal.max() - signal.min())
    p0 = [A0 * 0.6, 10.0, A0 * 0.4, 1.0, float(signal.min())]
    bounds = ([0, 0, 0, 0, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])

    try:
        popt, pcov = curve_fit(biexponential, t, signal, p0=p0, bounds=bounds, maxfev=10000)
        fitted = biexponential(t, *popt)
        residuals = signal - fitted
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((signal - signal.mean())**2))
        r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return {
            "A1": float(popt[0]), "k1": float(popt[1]),
            "A2": float(popt[2]), "k2": float(popt[3]),
            "C": float(popt[4]),
            "r_squared": r_squared,
            "fitted_curve": fitted,
            "success": True,
        }
    except RuntimeError as e:
        return {"success": False, "error": str(e)}

# Parameter sweep

def parameter_sweep(
    base_params: SimulationParams,
    sweep_params: dict[str, list],
) -> list[dict]:
    """
    Run a grid sweep over specified parameters and collect bi-exponential
    fit results for QH2 flux time courses.

    Parameters
    ----------
    base_params : SimulationParams
        Default simulation parameters.
    sweep_params : dict
        Mapping from parameter name to list of values to sweep over.
        E.g. {"D_ubiquinone": [0.1, 0.5, 1.0], "k_CIII_bind": [100, 200, 400]}

    Returns
    -------
    List of result dicts, one per parameter combination.
    """
    import copy

    param_names = list(sweep_params.keys())
    param_values = list(sweep_params.values())
    results = []

    for combo in itertools.product(*param_values):
        params = copy.deepcopy(base_params)
        combo_dict = dict(zip(param_names, combo))
        for pname, pval in combo_dict.items():
            setattr(params, pname, pval)

        sim = ETCSimulator(params)
        output = sim.run()
        fit = fit_biexponential(output["t"], output["QH2_counts"].astype(float))

        results.append({
            "params": combo_dict,
            "fit": {k: v for k, v in fit.items() if k != "fitted_curve"},
        })
        print(f"  Sweep {combo_dict} → k1={fit.get('k1', 'N/A'):.3f}, k2={fit.get('k2', 'N/A'):.3f}, R²={fit.get('r_squared', 'N/A')}")

    return results

# Entry point

if __name__ == "__main__":
    params = SimulationParams(
        grid_rows=40,
        grid_cols=40,
        t_end=0.2,
        seed=42,
    )
    sim = ETCSimulator(params)
    output = sim.run()

    print("\nBi-exponential fit to QH2 time course:")
    fit = fit_biexponential(output["t"], output["QH2_counts"].astype(float))
    if fit["success"]:
        print(f"  A1={fit['A1']:.2f}, k1={fit['k1']:.3f} s⁻¹")
        print(f"  A2={fit['A2']:.2f}, k2={fit['k2']:.3f} s⁻¹")
        print(f"  R²={fit['r_squared']:.4f}")
    else:
        print(f"  Fit failed: {fit['error']}")
