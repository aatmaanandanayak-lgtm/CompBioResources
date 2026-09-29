"""
in Matlab originally
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import pearsonr

# Bi-exponential model and fitting

def biexponential(t: np.ndarray, A1: float, k1: float, A2: float, k2: float, C: float) -> np.ndarray:
    return A1 * np.exp(-k1 * t) + A2 * np.exp(-k2 * t) + C


def fit_biexponential(
    t: np.ndarray,
    signal: np.ndarray,
    p0: Optional[list] = None,
) -> dict:
    """
    Fit a bi-exponential decay/rise model to a carrier count time course.

    The model is:
        f(t) = A1·exp(-k1·t) + A2·exp(-k2·t) + C

    where k1 > k2 by convention (fast and slow components).

    Returns a dict of fitted parameters, R², and the residuals.
    """
    if p0 is None:
        span = float(signal.max() - signal.min())
        p0 = [span * 0.6, 20.0, span * 0.4, 2.0, float(signal.min())]

    bounds = ([0, 1e-6, 0, 1e-6, -np.inf], [np.inf, np.inf, np.inf, np.inf, np.inf])
    try:
        popt, pcov = curve_fit(
            biexponential, t, signal.astype(float),
            p0=p0, bounds=bounds, maxfev=20000,
        )
        A1, k1, A2, k2, C = popt
        # Enforce k1 >= k2 convention (fast component first)
        if k1 < k2:
            A1, k1, A2, k2 = A2, k2, A1, k1

        fitted = biexponential(t, A1, k1, A2, k2, C)
        residuals = signal.astype(float) - fitted
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum((signal - signal.mean())**2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

        perr = np.sqrt(np.diag(pcov))
        return {
            "success": True,
            "A1": A1, "k1": k1, "A2": A2, "k2": k2, "C": C,
            "A1_err": perr[0], "k1_err": perr[1],
            "A2_err": perr[2], "k2_err": perr[3],
            "r_squared": r2,
            "residuals": residuals,
            "fitted": fitted,
        }
    except (RuntimeError, ValueError) as e:
        return {"success": False, "error": str(e)}

# Artefact correction via control runs


def correct_for_artefacts(
    signal: np.ndarray,
    control_signal: np.ndarray,
) -> np.ndarray:
    """
    Subtract simulation-software-introduced artefacts from a signal using
    a matched control run (same parameters, reactions disabled).

    The correction assumes that any systematic drift present in the control
    is attributable to numerical artefacts rather than biology, and subtracts
    it linearly from the experimental signal.

    Parameters
    ----------
    signal : np.ndarray
        Raw time-course from the experimental run.
    control_signal : np.ndarray
        Time-course from the corresponding control run (reactions disabled).

    Returns
    -------
    corrected : np.ndarray
        Artefact-corrected signal.
    """
    if len(signal) != len(control_signal):
        raise ValueError("Signal and control must have the same length.")

    # Normalise control to zero mean drift
    control_drift = control_signal - control_signal[0]
    corrected = signal - control_drift
    return corrected

# Plotting utilities


def plot_timecourse_with_fit(
    t: np.ndarray,
    signal: np.ndarray,
    fit_result: dict,
    species_name: str = "QH₂",
    output_path: Optional[str] = None,
) -> None:
    """Plot raw time course alongside bi-exponential fit and residuals."""
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]})

    ax = axes[0]
    ax.plot(t, signal, color="steelblue", lw=1.5, label="Simulation", alpha=0.8)
    if fit_result["success"]:
        ax.plot(t, fit_result["fitted"], color="tomato", lw=2.0,
                linestyle="--", label="Bi-exponential fit")
        ax.set_title(
            f"{species_name} — "
            f"k₁={fit_result['k1']:.2f} s⁻¹, k₂={fit_result['k2']:.2f} s⁻¹, "
            f"R²={fit_result['r_squared']:.4f}"
        )
    ax.set_ylabel("Molecule count")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)

    ax2 = axes[1]
    if fit_result["success"]:
        ax2.plot(t, fit_result["residuals"], color="grey", lw=1.0)
        ax2.axhline(0, color="black", lw=0.8, linestyle="--")
        ax2.set_ylabel("Residual")
    ax2.set_xlabel("Time (s)")
    ax2.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {output_path}")
    plt.show()


def plot_sweep_heatmap(
    sweep_results: list[dict],
    param_x: str,
    param_y: str,
    metric: str = "k1",
    output_path: Optional[str] = None,
) -> None:
    """
    Visualise a 2D parameter sweep as a heatmap.

    Parameters
    ----------
    sweep_results : list of dicts from parameter_sweep()
    param_x, param_y : parameter names for axes
    metric : which fitted quantity to show (e.g. "k1", "k2", "r_squared")
    """
    x_vals = sorted(set(r["params"][param_x] for r in sweep_results))
    y_vals = sorted(set(r["params"][param_y] for r in sweep_results))

    Z = np.full((len(y_vals), len(x_vals)), np.nan)
    x_idx = {v: i for i, v in enumerate(x_vals)}
    y_idx = {v: i for i, v in enumerate(y_vals)}

    for r in sweep_results:
        xi = x_idx[r["params"][param_x]]
        yi = y_idx[r["params"][param_y]]
        Z[yi, xi] = r["fit"].get(metric, np.nan)

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(Z, aspect="auto", origin="lower",
                   extent=[min(x_vals), max(x_vals), min(y_vals), max(y_vals)],
                   cmap="viridis")
    plt.colorbar(im, ax=ax, label=metric)
    ax.set_xlabel(param_x)
    ax.set_ylabel(param_y)
    ax.set_title(f"Parameter sweep: {metric}")
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to {output_path}")
    plt.show()

# Sensitivity index computation (Sobol-style rank correlation)

def compute_rank_sensitivity(
    sweep_results: list[dict],
    output_metric: str = "k1",
) -> dict[str, float]:
    """
    Estimate parameter sensitivity by Spearman rank correlation between
    each swept parameter and a chosen output metric across the sweep.

    Returns
    -------
    dict mapping parameter name → Spearman correlation coefficient
    """
    from scipy.stats import spearmanr

    if not sweep_results:
        return {}

    param_names = list(sweep_results[0]["params"].keys())
    sensitivities: dict[str, float] = {}

    outputs = np.array([r["fit"].get(output_metric, np.nan) for r in sweep_results])
    valid = ~np.isnan(outputs)

    for pname in param_names:
        inputs = np.array([r["params"][pname] for r in sweep_results])
        if valid.sum() < 3:
            sensitivities[pname] = float("nan")
            continue
        corr, pval = spearmanr(inputs[valid], outputs[valid])
        sensitivities[pname] = float(corr)

    # Sort by absolute sensitivity
    sensitivities = dict(
        sorted(sensitivities.items(), key=lambda kv: abs(kv[1]), reverse=True)
    )
    return sensitivities

# Entry point (example analysis on saved simulation output)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyse ETC simulation output.")
    parser.add_argument("--results_json", required=True,
                        help="Path to JSON file with simulation time courses.")
    parser.add_argument("--output_dir", default="figures/")
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    with open(args.results_json) as fh:
        data = json.load(fh)

    t = np.array(data["t"])
    qh2 = np.array(data["QH2_counts"])
    cytc_red = np.array(data["CytC_red_counts"])

    print("Fitting QH₂ time course ...")
    fit_qh2 = fit_biexponential(t, qh2)
    if fit_qh2["success"]:
        print(f"  k1 = {fit_qh2['k1']:.3f} ± {fit_qh2['k1_err']:.3f} s⁻¹")
        print(f"  k2 = {fit_qh2['k2']:.3f} ± {fit_qh2['k2_err']:.3f} s⁻¹")
        print(f"  R² = {fit_qh2['r_squared']:.4f}")
    else:
        print(f"  Fit failed: {fit_qh2['error']}")

    plot_timecourse_with_fit(
        t, qh2, fit_qh2,
        species_name="QH₂",
        output_path=str(Path(args.output_dir) / "qh2_fit.png"),
    )

    print("\nFitting CytC_red time course ...")
    fit_cytc = fit_biexponential(t, cytc_red)
    if fit_cytc["success"]:
        print(f"  k1 = {fit_cytc['k1']:.3f} ± {fit_cytc['k1_err']:.3f} s⁻¹")
        print(f"  k2 = {fit_cytc['k2']:.3f} ± {fit_cytc['k2_err']:.3f} s⁻¹")
        print(f"  R² = {fit_cytc['r_squared']:.4f}")
