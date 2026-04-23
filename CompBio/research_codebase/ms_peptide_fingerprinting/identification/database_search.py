"""
ms_peptide_fingerprinting/identification/database_search.py

Database search pipeline for peptide mass fingerprinting.

Given a set of MS/MS spectra and a protein sequence database (FASTA), this
module:

  1. Digests protein sequences in silico using a specified protease.
  2. Filters candidate peptides by precursor mass matching.
  3. Generates theoretical fragment spectra for each candidate.
  4. Scores candidates against observed spectra (hyperscore + cosine).
  5. Applies a target-decoy strategy for false discovery rate (FDR) estimation.
  6. Returns peptide-spectrum matches (PSMs) ranked by score.

Supported proteases: trypsin, Lys-C, Asp-N, Glu-C, chymotrypsin.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np

from ms_peptide_fingerprinting.fragmentation.fragment_ions import (
    TheoreticalFragmentGenerator,
    peptide_mass,
    neutral_mass_from_mz,
    Modification,
    RESIDUE_MASSES,
    H2O,
    PROTON,
)
from ms_peptide_fingerprinting.io.spectrum_io import MSMSSpectrum
from ms_peptide_fingerprinting.scoring.spectrum_scoring import (
    ScoringParams,
    hyperscore,
    cosine_score,
)


# ---------------------------------------------------------------------------
# Protease definitions (cleavage rules as regex)
# ---------------------------------------------------------------------------

PROTEASE_RULES: dict[str, str] = {
    "trypsin":        r"(?<=[KR])(?!P)",         # cleave after K/R, not before P
    "trypsin_strict": r"(?<=[KR])",               # strict trypsin (cleave after K/R)
    "lys_c":          r"(?<=K)",                  # cleave after K
    "asp_n":          r"(?=D)",                   # cleave before D
    "glu_c":          r"(?<=[DE])",               # cleave after D/E
    "chymotrypsin":   r"(?<=[FYWL])(?!P)",        # cleave after F/Y/W/L, not before P
    "no_enzyme":      r"(?<=.)",                  # all peptides (very slow)
}


# ---------------------------------------------------------------------------
# FASTA reader
# ---------------------------------------------------------------------------

def parse_fasta(filepath: str, decoy_prefix: str = "DECOY_") -> Iterator[tuple[str, str]]:
    """
    Parse a FASTA file and yield (header, sequence) tuples.

    If `decoy_prefix` is provided, also yields reversed decoy sequences for
    target-decoy FDR estimation.

    Parameters
    ----------
    filepath : str
        Path to .fasta or .fa file.
    decoy_prefix : str
        Prefix added to decoy sequence headers. Set to "" to skip decoys.
    """
    current_header = ""
    current_seq: list[str] = []

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if line.startswith(">"):
                if current_header and current_seq:
                    seq = "".join(current_seq)
                    yield current_header, seq
                    if decoy_prefix:
                        yield decoy_prefix + current_header, seq[::-1]
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line.upper().replace("*", "").replace("-", ""))

    if current_header and current_seq:
        seq = "".join(current_seq)
        yield current_header, seq
        if decoy_prefix:
            yield decoy_prefix + current_header, seq[::-1]


# ---------------------------------------------------------------------------
# In silico digestion
# ---------------------------------------------------------------------------

def digest_protein(
    sequence: str,
    protease: str = "trypsin",
    missed_cleavages: int = 2,
    min_length: int = 6,
    max_length: int = 50,
) -> list[str]:
    """
    Digest a protein sequence in silico and return the set of peptides.

    Parameters
    ----------
    sequence : str
        Protein amino acid sequence (one-letter codes, uppercase).
    protease : str
        Name of the protease. Must be a key in PROTEASE_RULES.
    missed_cleavages : int
        Maximum number of missed cleavage sites to allow.
    min_length, max_length : int
        Minimum/maximum peptide length to retain.

    Returns
    -------
    List of unique peptide strings.
    """
    rule = PROTEASE_RULES.get(protease)
    if rule is None:
        raise ValueError(f"Unknown protease '{protease}'. "
                         f"Choose from: {list(PROTEASE_RULES.keys())}")

    # Split sequence at cleavage sites
    parts = re.split(rule, sequence)

    # Filter by length
    parts = [p for p in parts if min_length <= len(p) <= max_length]

    # Generate missed-cleavage combinations
    peptides: set[str] = set()
    for i in range(len(parts)):
        for mc in range(missed_cleavages + 1):
            if i + mc + 1 <= len(parts):
                pep = "".join(parts[i: i + mc + 1])
                if min_length <= len(pep) <= max_length:
                    peptides.add(pep)

    return list(peptides)


# ---------------------------------------------------------------------------
# Peptide-spectrum match result
# ---------------------------------------------------------------------------

@dataclass
class PSM:
    """A peptide-spectrum match (PSM) from database search."""
    scan_id: str
    peptide: str
    protein: str
    precursor_mz: float
    precursor_charge: int
    hyperscore: float
    cosine_score: float
    n_matched_ions: int
    delta_mass: float           # observed - theoretical precursor mass (Da)
    is_decoy: bool = False
    fdr: float = float("nan")

    @property
    def combined_score(self) -> float:
        """Simple linear combination for ranking."""
        return self.hyperscore + 5.0 * self.cosine_score


# ---------------------------------------------------------------------------
# FDR estimation (target-decoy)
# ---------------------------------------------------------------------------

def estimate_fdr(psms: list[PSM], decoy_prefix: str = "DECOY_") -> list[PSM]:
    """
    Estimate PSM-level FDR using the target-decoy approach.

    PSMs are sorted by descending combined score. At each score threshold,
    FDR is estimated as:

        FDR(t) = 2 · #decoys(score ≥ t) / #total(score ≥ t)

    (factor of 2 because decoy database is same size as target database.)

    The estimated FDR is assigned back to each PSM and the list is returned
    sorted by score.
    """
    for psm in psms:
        psm.is_decoy = psm.protein.startswith(decoy_prefix)

    psms_sorted = sorted(psms, key=lambda p: p.combined_score, reverse=True)

    decoy_count = 0
    for i, psm in enumerate(psms_sorted):
        if psm.is_decoy:
            decoy_count += 1
        total = i + 1
        psm.fdr = min(2.0 * decoy_count / total, 1.0)

    return psms_sorted


# ---------------------------------------------------------------------------
# Main database search engine
# ---------------------------------------------------------------------------

class DatabaseSearchEngine:
    """
    Peptide identification by database search against a FASTA file.

    Parameters
    ----------
    fasta_path : str
        Path to protein sequence database in FASTA format.
    protease : str
        Protease for in silico digestion.
    missed_cleavages : int
        Maximum missed cleavages.
    precursor_tol_ppm : float
        Precursor mass tolerance for candidate filtering (ppm).
    scoring_params : ScoringParams
        Fragment matching and scoring parameters.
    fixed_mods : list of Modification
        Fixed modifications applied to all instances of a residue.
    variable_mods : list of Modification
        Variable modifications — generates modified peptide variants.
    decoy_prefix : str
        Prefix for decoy sequences in FDR estimation.
    """

    def __init__(
        self,
        fasta_path: str,
        protease: str = "trypsin",
        missed_cleavages: int = 2,
        precursor_tol_ppm: float = 10.0,
        scoring_params: Optional[ScoringParams] = None,
        fixed_mods: Optional[list[Modification]] = None,
        variable_mods: Optional[list[Modification]] = None,
        decoy_prefix: str = "DECOY_",
    ) -> None:
        self.fasta_path = fasta_path
        self.protease = protease
        self.missed_cleavages = missed_cleavages
        self.precursor_tol_ppm = precursor_tol_ppm
        self.scoring_params = scoring_params or ScoringParams()
        self.fixed_mods = fixed_mods or []
        self.variable_mods = variable_mods or []
        self.decoy_prefix = decoy_prefix

        self._fragment_gen = TheoreticalFragmentGenerator(
            ion_types=self.scoring_params.ion_types,
            max_charge=self.scoring_params.max_fragment_charge,
            neutral_losses=self.scoring_params.neutral_losses,
        )

        # Build peptide database: {peptide_seq: [protein_header, ...]}
        print(f"Building peptide database from {fasta_path} ...")
        self._peptide_db = self._build_database()
        print(f"  {len(self._peptide_db)} unique peptides indexed.")

    def _build_database(self) -> dict[str, list[str]]:
        """Digest all proteins and build a peptide → protein mapping."""
        db: dict[str, list[str]] = {}
        for header, protein_seq in parse_fasta(self.fasta_path, self.decoy_prefix):
            peptides = digest_protein(
                protein_seq,
                protease=self.protease,
                missed_cleavages=self.missed_cleavages,
            )
            for pep in peptides:
                db.setdefault(pep, []).append(header)
        return db

    def search_spectrum(self, spectrum: MSMSSpectrum) -> list[PSM]:
        """
        Search a single MS/MS spectrum against the peptide database.

        Returns a list of PSMs sorted by combined score (descending).
        """
        precursor_neutral = neutral_mass_from_mz(
            spectrum.precursor_mz, max(spectrum.precursor_charge, 1)
        )
        tol_da = precursor_neutral * self.precursor_tol_ppm / 1e6

        psms: list[PSM] = []

        for peptide, proteins in self._peptide_db.items():
            # Apply fixed modifications
            mods = list(self.fixed_mods)
            pep_mass = peptide_mass(peptide, mods, charge=1) * 1 - PROTON

            # Filter by precursor mass
            if abs(pep_mass - precursor_neutral) > tol_da:
                continue

            # Generate theoretical ions
            theo_ions = self._fragment_gen.generate(peptide, mods)

            # Score
            hs = hyperscore(spectrum, theo_ions, self.scoring_params)
            cs = cosine_score(spectrum, theo_ions, self.scoring_params)

            if hs == 0.0 and cs < 0.01:
                continue

            # Count matched ions
            from ms_peptide_fingerprinting.scoring.spectrum_scoring import match_fragments
            matches = match_fragments(
                spectrum.mz, spectrum.intensity, theo_ions,
                self.scoring_params.fragment_tol_da,
            )

            for protein in proteins:
                psms.append(PSM(
                    scan_id=spectrum.scan_id,
                    peptide=peptide,
                    protein=protein,
                    precursor_mz=spectrum.precursor_mz,
                    precursor_charge=spectrum.precursor_charge,
                    hyperscore=hs,
                    cosine_score=cs,
                    n_matched_ions=len(matches),
                    delta_mass=precursor_neutral - pep_mass,
                    is_decoy=protein.startswith(self.decoy_prefix),
                ))

        psms.sort(key=lambda p: p.combined_score, reverse=True)
        return psms

    def search_all(
        self,
        spectra: list[MSMSSpectrum],
        top_n_per_spectrum: int = 5,
        fdr_threshold: float = 0.01,
    ) -> list[PSM]:
        """
        Search a list of spectra and return filtered, FDR-controlled PSMs.

        Parameters
        ----------
        spectra : list of MSMSSpectrum
        top_n_per_spectrum : int
            Number of top PSMs to retain per spectrum before FDR filtering.
        fdr_threshold : float
            Maximum FDR to report (e.g. 0.01 = 1%).

        Returns
        -------
        List of PSMs at the specified FDR level.
        """
        all_psms: list[PSM] = []
        for i, spectrum in enumerate(spectra):
            if (i + 1) % 100 == 0:
                print(f"  Searching spectrum {i + 1}/{len(spectra)} ...")
            psms = self.search_spectrum(spectrum)
            all_psms.extend(psms[:top_n_per_spectrum])

        print(f"  {len(all_psms)} PSMs before FDR filtering.")
        all_psms = estimate_fdr(all_psms, self.decoy_prefix)
        passing = [p for p in all_psms if p.fdr <= fdr_threshold and not p.is_decoy]
        print(f"  {len(passing)} PSMs pass FDR ≤ {fdr_threshold * 100:.1f}%.")
        return passing
