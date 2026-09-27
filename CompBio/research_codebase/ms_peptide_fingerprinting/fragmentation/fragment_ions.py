"""
ms_peptide_fingerprinting/fragmentation/fragment_ions.py

Background 
For a peptide of sequence P₁P₂…Pₙ, collision-induced dissociation (CID) and
higher-energy collisional dissociation (HCD) produce two primary ion series:

  b-ions : N-terminal fragments  [P₁…Pᵢ + H]⁺   (i = 1 … n-1)
  y-ions : C-terminal fragments  [Pᵢ₊₁…Pₙ + H + OH]⁺  (i = 1 … n-1)

Additional series (a, c, x, z) are also generated for ETD/ECD fragmentation.

Each ion series is calculated from residue monoisotopic masses plus the
relevant terminus and charge-carrier corrections. Variable and fixed
modifications (e.g. carbamidomethylation of Cys, phosphorylation) shift
the relevant residue masses before the series is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

# Monoisotopic residue masses (Da)
# Standard 20 amino acids

RESIDUE_MASSES: dict[str, float] = {
    "A": 71.03711,
    "R": 156.10111,
    "N": 114.04293,
    "D": 115.02694,
    "C": 103.00919,
    "E": 129.04259,
    "Q": 128.05858,
    "G": 57.02146,
    "H": 137.05891,
    "I": 113.08406,
    "L": 113.08406,
    "K": 128.09496,
    "M": 131.04049,
    "F": 147.06841,
    "P": 97.05276,
    "S": 87.03203,
    "T": 101.04768,
    "W": 186.07931,
    "Y": 163.06333,
    "V": 99.06841,
}

# Common modification mass shifts (Da)
MODIFICATION_MASSES: dict[str, float] = {
    "Carbamidomethyl(C)": 57.02146,   # fixed Cys alkylation
    "Oxidation(M)": 15.99491,          # Met oxidation
    "Phospho(S)": 79.96633,            # Ser phosphorylation
    "Phospho(T)": 79.96633,            # Thr phosphorylation
    "Phospho(Y)": 79.96633,            # Tyr phosphorylation
    "Acetyl(K)": 42.01057,             # Lys acetylation
    "Deamidation(N)": 0.98402,         # Asn deamidation
    "Deamidation(Q)": 0.98402,         # Gln deamidation
    "GlyGly(K)": 114.04293,            # Lys ubiquitination remnant
}

PROTON = 1.007276
H2O = 18.010565
NH3 = 17.026549
CO = 27.994915
H = 1.007825

# Modification specification

@dataclass
class Modification:
    """
    A post-translational or chemical modification applied to a specific residue.

    Attributes
    ----------
    position : int
        0-based position in the peptide sequence (None = N/C terminus).
    name : str
        Modification name (must be a key in MODIFICATION_MASSES or provide delta_mass).
    delta_mass : float
        Mass shift in Da. Looked up from MODIFICATION_MASSES if not provided.
    """
    position: int
    name: str
    delta_mass: Optional[float] = None

    def __post_init__(self) -> None:
        if self.delta_mass is None:
            if self.name not in MODIFICATION_MASSES:
                raise ValueError(
                    f"Unknown modification '{self.name}'. "
                    f"Provide delta_mass explicitly or add to MODIFICATION_MASSES."
                )
            self.delta_mass = MODIFICATION_MASSES[self.name]

# Fragment ion data container

@dataclass
class FragmentIon:
    """A single theoretical fragment ion."""
    ion_type: str      # "b", "y", "a", "c", "x", "z"
    ion_number: int    # position index (1-based)
    charge: int        # charge state
    mz: float          # computed m/z value
    sequence: str      # amino acid sequence of the fragment

# Theoretical spectrum generator

class TheoreticalFragmentGenerator:
    """
    Generates theoretical CID/HCD/ETD fragment ion series for a peptide sequence.

    Parameters
    ----------
    ion_types : list of str
        Which ion series to generate. Subset of {"b", "y", "a", "c", "x", "z"}.
        Default: ["b", "y"] (standard CID/HCD).
    max_charge : int
        Maximum fragment charge state to generate. Default 2.
    neutral_losses : bool
        Whether to include common neutral-loss peaks (NH3, H2O) for b/y ions.
    """

    def __init__(
        self,
        ion_types: list[str] = None,
        max_charge: int = 2,
        neutral_losses: bool = True,
    ) -> None:
        self.ion_types = ion_types if ion_types is not None else ["b", "y"]
        self.max_charge = max_charge
        self.neutral_losses = neutral_losses

    def generate(
        self,
        sequence: str,
        modifications: Optional[list[Modification]] = None,
    ) -> list[FragmentIon]:
        """
        Generate all theoretical fragment ions for a peptide.

        Parameters
        ----------
        sequence : str
            One-letter amino acid sequence (uppercase).
        modifications : list of Modification, optional
            Modifications to apply before computing fragment masses.

        Returns
        -------
        list of FragmentIon objects sorted by m/z.
        """
        sequence = sequence.upper()
        n = len(sequence)

        # Build array of modified residue masses
        residue_masses = self._build_residue_masses(sequence, modifications or [])

        ions: list[FragmentIon] = []

        for ion_type in self.ion_types:
            ions.extend(self._generate_series(sequence, residue_masses, ion_type))

        ions.sort(key=lambda ion: ion.mz)
        return ions

    def _build_residue_masses(
        self,
        sequence: str,
        modifications: list[Modification],
    ) -> np.ndarray:
        """Return per-residue masses with modifications applied."""
        masses = np.array(
            [RESIDUE_MASSES.get(aa, RESIDUE_MASSES["G"]) for aa in sequence],
            dtype=np.float64,
        )
        for mod in modifications:
            if 0 <= mod.position < len(masses):
                masses[mod.position] += mod.delta_mass
        return masses

    def _generate_series(
        self,
        sequence: str,
        residue_masses: np.ndarray,
        ion_type: str,
    ) -> list[FragmentIon]:
        """Generate all ions of a given type."""
        n = len(sequence)
        ions: list[FragmentIon] = []

        prefix_masses = np.cumsum(residue_masses)   # cumulative mass from N-terminus
        suffix_masses = np.cumsum(residue_masses[::-1])[::-1]  # from C-terminus

        for i in range(1, n):
            for z in range(1, self.max_charge + 1):
                mz = self._compute_mz(
                    prefix_masses, suffix_masses, ion_type, i, n, z
                )
                if mz is None or mz <= 0:
                    continue

                frag_seq = self._fragment_sequence(sequence, ion_type, i, n)
                ions.append(FragmentIon(
                    ion_type=ion_type,
                    ion_number=i,
                    charge=z,
                    mz=mz,
                    sequence=frag_seq,
                ))

                # Neutral losses
                if self.neutral_losses and ion_type in ("b", "y"):
                    for loss_name, loss_mass in [("H2O", H2O), ("NH3", NH3)]:
                        mz_nl = (mz * z - loss_mass) / z
                        if mz_nl > 0:
                            ions.append(FragmentIon(
                                ion_type=f"{ion_type}-{loss_name}",
                                ion_number=i,
                                charge=z,
                                mz=mz_nl,
                                sequence=frag_seq,
                            ))

        return ions

    def _compute_mz(
        self,
        prefix_masses: np.ndarray,
        suffix_masses: np.ndarray,
        ion_type: str,
        i: int,
        n: int,
        z: int,
    ) -> Optional[float]:
        """
        Compute the m/z for a given ion type, cleavage position, and charge.

        Ion mass formulae (singly charged, then divide by z):
          b[i] = sum(residues 1..i) + H⁺
          y[i] = sum(residues n-i+1..n) + H₂O + H⁺
          a[i] = b[i] - CO
          c[i] = b[i] + NH₃
          x[i] = y[i] + CO - H₂
          z[i] = y[i] - NH₃
        """
        prefix = prefix_masses[i - 1]   # sum of first i residues
        suffix = suffix_masses[n - i]   # sum of last (n-i) residues

        if ion_type == "b":
            neutral = prefix
        elif ion_type == "y":
            neutral = suffix + H2O
        elif ion_type == "a":
            neutral = prefix - CO
        elif ion_type == "c":
            neutral = prefix + NH3
        elif ion_type == "x":
            neutral = suffix + H2O + CO - 2 * H
        elif ion_type == "z":
            neutral = suffix + H2O - NH3
        else:
            return None

        return (neutral + z * PROTON) / z

    @staticmethod
    def _fragment_sequence(sequence: str, ion_type: str, i: int, n: int) -> str:
        """Return the subsequence corresponding to a fragment."""
        if ion_type in ("b", "a", "c"):
            return sequence[:i]
        else:  # y, x, z
            return sequence[n - i:]

# Peptide mass calculator

def peptide_mass(
    sequence: str,
    modifications: Optional[list[Modification]] = None,
    charge: int = 1,
) -> float:
    """
    Compute the monoisotopic m/z of a peptide at a given charge state.

    M = sum(residue masses) + H₂O   (neutral mass)
    m/z = (M + z * proton) / z
    """
    masses = np.array(
        [RESIDUE_MASSES.get(aa, RESIDUE_MASSES["G"]) for aa in sequence.upper()],
        dtype=np.float64,
    )
    if modifications:
        for mod in modifications:
            if 0 <= mod.position < len(masses):
                masses[mod.position] += mod.delta_mass

    neutral_mass = masses.sum() + H2O
    return (neutral_mass + charge * PROTON) / charge


def neutral_mass_from_mz(mz: float, charge: int) -> float:
    """Convert precursor m/z and charge to neutral monoisotopic mass."""
    return mz * charge - charge * PROTON
