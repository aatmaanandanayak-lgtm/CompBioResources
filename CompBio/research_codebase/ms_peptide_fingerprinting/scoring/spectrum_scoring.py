"""
ms_peptide_fingerprinting/scoring/spectrum_scoring.py

Scoring functions for matching observed MS/MS spectra against theoretical
fragment ion series.

Two complementary approaches are implemented:

1. Database search scoring
   For each candidate peptide from a sequence database, generate its
   theoretical spectrum and score it against the observed spectrum.
   Score functions implemented:
     - Hyperscore (X!Tandem-style): product of matched b/y ion counts
       and summed matched intensities, log-transformed.
     - Dot-product cosine similarity: normalised dot product between
       observed and theoretical intensity vectors.

2. De novo sequencing
   Interprets the spectrum directly without a database by finding the
   highest-scoring path through a directed acyclic graph (DAG) whose
   nodes are observed peaks and whose edges correspond to amino acid
   residue mass differences.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ms_peptide_fingerprinting.fragmentation.fragment_ions import (
    FragmentIon,
    TheoreticalFragmentGenerator,
    RESIDUE_MASSES,
    PROTON,
    H2O,
)
from ms_peptide_fingerprinting.io.spectrum_io import MSMSSpectrum


# ---------------------------------------------------------------------------
# Scoring parameters
# ---------------------------------------------------------------------------

@dataclass
class ScoringParams:
    """Parameters controlling fragment matching and scoring."""
    fragment_tol_da: float = 0.02      # fragment mass tolerance (Da)
    fragment_tol_ppm: float = 0.0      # fragment mass tolerance (ppm); if > 0, overrides Da
    min_matched_ions: int = 4          # minimum matched ions to report a hit
    ion_types: list = field(default_factory=lambda: ["b", "y"])
    max_fragment_charge: int = 2
    neutral_losses: bool = True


# ---------------------------------------------------------------------------
# Fragment matching
# ---------------------------------------------------------------------------

def match_fragments(
    observed_mz: np.ndarray,
    observed_intensity: np.ndarray,
    theoretical_ions: list[FragmentIon],
    tol_da: float = 0.02,
    tol_ppm: float = 0.0,
) -> list[tuple[FragmentIon, float]]:
    """
    Match theoretical fragment ions to observed peaks within a mass tolerance.

    For each theoretical ion, the closest observed peak within tolerance is
    selected. If multiple theoretical ions map to the same observed peak,
    the closest one wins.

    Parameters
    ----------
    observed_mz : np.ndarray (N,)
    observed_intensity : np.ndarray (N,)
    theoretical_ions : list of FragmentIon
    tol_da : float
        Absolute tolerance in Daltons. Used if tol_ppm == 0.
    tol_ppm : float
        Relative tolerance in ppm. If > 0, overrides tol_da.

    Returns
    -------
    List of (FragmentIon, matched_intensity) tuples for each matched ion.
    """
    matches: list[tuple[FragmentIon, float]] = []
    used_obs_indices: set[int] = set()

    for ion in theoretical_ions:
        # Compute tolerance for this ion
        if tol_ppm > 0:
            tol = ion.mz * tol_ppm / 1e6
        else:
            tol = tol_da

        diffs = np.abs(observed_mz - ion.mz)
        candidate_mask = diffs <= tol

        if not candidate_mask.any():
            continue

        # Among candidates, pick closest unused peak
        candidate_indices = np.where(candidate_mask)[0]
        candidate_diffs = diffs[candidate_indices]
        order = np.argsort(candidate_diffs)

        for idx_in_candidates in order:
            obs_idx = candidate_indices[idx_in_candidates]
            if obs_idx not in used_obs_indices:
                used_obs_indices.add(obs_idx)
                matches.append((ion, float(observed_intensity[obs_idx])))
                break

    return matches


# ---------------------------------------------------------------------------
# Hyperscore (X!Tandem-style)
# ---------------------------------------------------------------------------

def hyperscore(
    observed: MSMSSpectrum,
    theoretical_ions: list[FragmentIon],
    params: ScoringParams,
) -> float:
    """
    Compute the X!Tandem hyperscore for a peptide–spectrum match.

    Hyperscore = log( Nb! · Ny! · Σ(Ib) · Σ(Iy) )

    where Nb, Ny are the counts of matched b and y ions respectively, and
    Σ(Ib), Σ(Iy) are the summed matched intensities for each series.

    The factorial terms reward matching more ions; the intensity sums reward
    matching intense peaks.

    Returns 0.0 if fewer than `params.min_matched_ions` are matched.
    """
    matches = match_fragments(
        observed.mz, observed.intensity,
        theoretical_ions, params.fragment_tol_da, params.fragment_tol_ppm,
    )

    if len(matches) < params.min_matched_ions:
        return 0.0

    b_count = 0
    y_count = 0
    b_intensity_sum = 0.0
    y_intensity_sum = 0.0

    for ion, intensity in matches:
        if ion.ion_type == "b":
            b_count += 1
            b_intensity_sum += intensity
        elif ion.ion_type == "y":
            y_count += 1
            y_intensity_sum += intensity

    if b_intensity_sum == 0.0 or y_intensity_sum == 0.0:
        return 0.0

    # Factorial via log-gamma for numerical stability
    log_score = (
        math.lgamma(b_count + 1)
        + math.lgamma(y_count + 1)
        + math.log(b_intensity_sum + 1e-10)
        + math.log(y_intensity_sum + 1e-10)
    )
    return log_score


# ---------------------------------------------------------------------------
# Cosine similarity score
# ---------------------------------------------------------------------------

def cosine_score(
    observed: MSMSSpectrum,
    theoretical_ions: list[FragmentIon],
    params: ScoringParams,
    n_bins: int = 2000,
    mz_min: float = 100.0,
    mz_max: float = 2000.0,
) -> float:
    """
    Normalised dot-product (cosine) similarity between observed and theoretical
    spectra represented as binned intensity vectors.

    Score = (obs · theo) / (|obs| · |theo|)

    A score of 1.0 indicates a perfect match; 0.0 indicates no overlap.
    """
    bin_edges = np.linspace(mz_min, mz_max, n_bins + 1)
    bin_width = (mz_max - mz_min) / n_bins

    # Observed vector
    obs_vec = np.zeros(n_bins)
    for mz_val, int_val in zip(observed.mz, observed.intensity):
        idx = int((mz_val - mz_min) / bin_width)
        if 0 <= idx < n_bins:
            obs_vec[idx] += int_val

    # Theoretical vector (unit intensities for matched ions)
    theo_vec = np.zeros(n_bins)
    for ion in theoretical_ions:
        idx = int((ion.mz - mz_min) / bin_width)
        if 0 <= idx < n_bins:
            theo_vec[idx] += 1.0

    obs_norm = np.linalg.norm(obs_vec)
    theo_norm = np.linalg.norm(theo_vec)

    if obs_norm == 0.0 or theo_norm == 0.0:
        return 0.0

    return float(np.dot(obs_vec, theo_vec) / (obs_norm * theo_norm))


# ---------------------------------------------------------------------------
# De novo sequencing via DAG path search
# ---------------------------------------------------------------------------

@dataclass
class DeNovoNode:
    """A node in the de novo sequencing DAG, corresponding to an observed peak."""
    peak_idx: int     # index into the spectrum's mz/intensity arrays
    mz: float
    intensity: float
    prefix_mass: float   # inferred N-terminal prefix mass at this node


@dataclass
class DeNovoPath:
    """A completed de novo sequence path through the DAG."""
    sequence: str
    score: float
    total_intensity: float
    n_matched: int


class DeNovoSequencer:
    """
    De novo peptide sequence determination from an MS/MS spectrum.

    Algorithm overview
    ------------------
    1. Convert observed peaks to prefix masses:
         prefix_mass = mz * z - z * H⁺ - H₂O   (for b-ion interpretation)
    2. Build a directed acyclic graph (DAG):
         Nodes : peaks (sorted by prefix mass)
         Edges : pairs (i, j) where mass(j) - mass(i) ≈ a standard residue mass
    3. Find the highest-scoring path from a virtual start node (mass ≈ 0)
       to a virtual end node (mass ≈ precursor neutral mass).
    4. Read off the amino acid sequence from the edge labels along the path.

    Score per edge = intensity of the destination peak (rewards matching
    intense peaks). A small bonus is added for complementary b/y ion pairs.

    Parameters
    ----------
    fragment_tol_da : float
        Mass tolerance (Da) for edge creation and residue assignment.
    max_gap_aa : int
        Maximum number of amino acids to skip in a single edge (for handling
        missed or undetected fragment ions). Default 1 (no gaps).
    """

    def __init__(
        self,
        fragment_tol_da: float = 0.02,
        max_gap_aa: int = 1,
    ) -> None:
        self.fragment_tol_da = fragment_tol_da
        self.max_gap_aa = max_gap_aa

        # Build residue mass lookup: mass → list of aa symbols
        self._residue_masses = RESIDUE_MASSES
        self._mass_to_aa: list[tuple[float, str]] = sorted(
            self._residue_masses.items(), key=lambda kv: kv[1]
        )

    def sequence_spectrum(
        self,
        spectrum: MSMSSpectrum,
        top_n: int = 5,
    ) -> list[DeNovoPath]:
        """
        Perform de novo sequencing on a preprocessed spectrum.

        Parameters
        ----------
        spectrum : MSMSSpectrum
            Preprocessed spectrum (noise-filtered, normalised).
        top_n : int
            Number of top-scoring candidate sequences to return.

        Returns
        -------
        List of DeNovoPath objects sorted by descending score.
        """
        if spectrum.precursor_charge == 0:
            return []

        precursor_neutral = spectrum.precursor_mass
        mz = spectrum.mz
        intensity = spectrum.intensity
        n_peaks = len(mz)

        if n_peaks < 3:
            return []

        # Interpret peaks as b-ions: prefix_mass = mz - H⁺ (singly charged)
        prefix_masses = mz - PROTON   # assume z=1 for fragment ions

        # Add virtual start (mass 0) and end (precursor neutral - H2O) nodes
        start_mass = 0.0
        end_mass = precursor_neutral - H2O

        # Build DAG: for each peak pair, check if mass diff ≈ residue mass
        # dp[i] = (best_score, best_sequence, prev_idx)
        dp: dict[int, tuple[float, str, int]] = {}
        # Index -1 = start node, index n_peaks = end node
        dp[-1] = (0.0, "", -2)

        for j in range(n_peaks):
            best_score = -1.0
            best_seq = ""
            best_prev = -2

            # Check start → j
            delta = prefix_masses[j] - start_mass
            aa = self._mass_to_aa_symbol(delta)
            if aa:
                score = float(intensity[j])
                if score > best_score:
                    best_score = score
                    best_seq = aa
                    best_prev = -1

            # Check previous peaks → j
            for i in range(j):
                delta = prefix_masses[j] - prefix_masses[i]
                if delta < 50.0:
                    continue
                aa = self._mass_to_aa_symbol(delta)
                if aa:
                    prev_score = dp.get(i, (-1.0, "", -2))[0]
                    if prev_score < 0:
                        continue
                    score = prev_score + float(intensity[j])
                    if score > best_score:
                        best_score = score
                        best_seq = dp[i][1] + aa
                        best_prev = i

            if best_score >= 0:
                dp[j] = (best_score, best_seq, best_prev)

        # Collect paths reaching the end mass (precursor)
        paths: list[DeNovoPath] = []
        tol = self.fragment_tol_da

        for j, (score, seq, _) in dp.items():
            if j < 0:
                continue
            gap = abs(prefix_masses[j] - end_mass)
            if gap <= tol and len(seq) >= 4:
                paths.append(DeNovoPath(
                    sequence=seq,
                    score=score,
                    total_intensity=score,
                    n_matched=len(seq),
                ))

        # Deduplicate and sort
        seen: set[str] = set()
        unique_paths: list[DeNovoPath] = []
        for p in sorted(paths, key=lambda x: x.score, reverse=True):
            if p.sequence not in seen:
                seen.add(p.sequence)
                unique_paths.append(p)
            if len(unique_paths) >= top_n:
                break

        return unique_paths

    def _mass_to_aa_symbol(self, mass: float) -> Optional[str]:
        """Return the single-letter code of the residue whose mass matches."""
        tol = self.fragment_tol_da
        for aa, aa_mass in self._mass_to_aa:
            if abs(aa_mass - mass) <= tol:
                return aa
            if aa_mass - mass > tol + 5:
                break
        return None


from typing import Optional  # noqa: E402 — placed here to avoid circular at top
