"""
ms_peptide_fingerprinting/io/spectrum_io.py

Reading, parsing, and preprocessing of MS/MS spectra for peptide mass
fingerprinting and de novo sequencing pipelines.

Supported input formats:
  - MGF (Mascot Generic Format)     — the most common interchange format
  - mzML                            — the HUPO-PSI standard XML format
  - plain two-column text (m/z, intensity)

Preprocessing steps applied before database search or de novo sequencing:
  1. Noise filtering   — remove peaks below an absolute or relative threshold
  2. Deisotoping       — collapse isotope envelopes to monoisotopic peaks
  3. Charge deconvolution — convert multiply-charged peaks to singly-charged
  4. Normalisation     — scale intensities to [0, 1] or to base peak
  5. Peak picking      — retain top-N most intense peaks per spectrum
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data containers
# ---------------------------------------------------------------------------

@dataclass
class MSMSSpectrum:
    """
    A single MS/MS spectrum with associated metadata.

    Attributes
    ----------
    scan_id : str
        Unique scan identifier (from file header or sequential index).
    precursor_mz : float
        m/z of the selected precursor ion.
    precursor_charge : int
        Charge state of the precursor ion (0 if unknown).
    precursor_mass : float
        Neutral monoisotopic mass of the precursor (computed from mz and z).
    mz : np.ndarray
        Sorted array of fragment ion m/z values.
    intensity : np.ndarray
        Corresponding fragment ion intensities.
    rt : float
        Retention time (seconds). 0.0 if not available.
    title : str
        Optional free-text title field (MGF TITLE tag).
    """
    scan_id: str
    precursor_mz: float
    precursor_charge: int
    mz: np.ndarray
    intensity: np.ndarray
    rt: float = 0.0
    title: str = ""

    @property
    def precursor_mass(self) -> float:
        """Neutral monoisotopic mass from precursor m/z and charge."""
        proton = 1.007276
        z = max(self.precursor_charge, 1)
        return self.precursor_mz * z - z * proton

    @property
    def n_peaks(self) -> int:
        return len(self.mz)

    def __repr__(self) -> str:
        return (
            f"MSMSSpectrum(scan_id={self.scan_id!r}, "
            f"precursor_mz={self.precursor_mz:.4f}, "
            f"z={self.precursor_charge}, "
            f"n_peaks={self.n_peaks})"
        )


# ---------------------------------------------------------------------------
# MGF parser
# ---------------------------------------------------------------------------

def parse_mgf(filepath: str) -> Iterator[MSMSSpectrum]:
    """
    Parse an MGF file and yield MSMSSpectrum objects one at a time.

    The MGF format consists of blocks delimited by BEGIN IONS / END IONS.
    Each block may contain header key=value lines followed by m/z intensity
    peak lines.

    Parameters
    ----------
    filepath : str
        Path to the .mgf file.

    Yields
    ------
    MSMSSpectrum
        One spectrum per BEGIN IONS / END IONS block.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"MGF file not found: {filepath}")

    scan_counter = 0
    in_spectrum = False
    title = ""
    precursor_mz = 0.0
    precursor_charge = 0
    rt = 0.0
    mz_list: list[float] = []
    intensity_list: list[float] = []

    with open(filepath, "r") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line == "BEGIN IONS":
                in_spectrum = True
                title = ""
                precursor_mz = 0.0
                precursor_charge = 0
                rt = 0.0
                mz_list = []
                intensity_list = []
                scan_counter += 1
                continue

            if line == "END IONS":
                in_spectrum = False
                if mz_list:
                    mz_arr = np.array(mz_list, dtype=np.float64)
                    int_arr = np.array(intensity_list, dtype=np.float64)
                    order = np.argsort(mz_arr)
                    yield MSMSSpectrum(
                        scan_id=str(scan_counter),
                        precursor_mz=precursor_mz,
                        precursor_charge=precursor_charge,
                        mz=mz_arr[order],
                        intensity=int_arr[order],
                        rt=rt,
                        title=title,
                    )
                continue

            if not in_spectrum:
                continue

            # Header key=value lines
            if "=" in line and not _is_peak_line(line):
                key, _, value = line.partition("=")
                key = key.strip().upper()
                value = value.strip()
                if key == "TITLE":
                    title = value
                elif key == "PEPMASS":
                    parts = value.split()
                    precursor_mz = float(parts[0])
                elif key == "CHARGE":
                    charge_str = value.replace("+", "").replace("-", "")
                    try:
                        precursor_charge = int(charge_str)
                    except ValueError:
                        precursor_charge = 0
                elif key in ("RTINSECONDS", "RT"):
                    try:
                        rt = float(value)
                    except ValueError:
                        rt = 0.0
            else:
                # Peak line: m/z [intensity]
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz_list.append(float(parts[0]))
                        intensity_list.append(float(parts[1]))
                    except ValueError:
                        pass
                elif len(parts) == 1:
                    try:
                        mz_list.append(float(parts[0]))
                        intensity_list.append(1.0)
                    except ValueError:
                        pass


def _is_peak_line(line: str) -> bool:
    """Return True if a line looks like an m/z [intensity] peak rather than key=value."""
    parts = line.split()
    if not parts:
        return False
    try:
        float(parts[0])
        return True
    except ValueError:
        return False


# ---------------------------------------------------------------------------
# Simple mzML parser (minimal, schema-independent)
# ---------------------------------------------------------------------------

def parse_mzml(filepath: str) -> Iterator[MSMSSpectrum]:
    """
    Parse an mzML file and yield MS2 spectra as MSMSSpectrum objects.

    Uses Python's built-in xml.etree.ElementTree for parsing; no external
    dependencies required. Handles base64-encoded binary peak arrays.

    Parameters
    ----------
    filepath : str
        Path to the .mzML file.
    """
    import base64
    import struct
    import xml.etree.ElementTree as ET
    import zlib

    NS = "http://psi.hupo.org/ms/mzml"

    def decode_array(encoded: str, compression: str, dtype_str: str) -> np.ndarray:
        raw = base64.b64decode(encoded)
        if compression == "zlib":
            raw = zlib.decompress(raw)
        dtype = np.float64 if "64" in dtype_str else np.float32
        arr = np.frombuffer(raw, dtype=dtype).copy()
        return arr.astype(np.float64)

    tree = ET.parse(filepath)
    root = tree.getroot()

    def tag(name: str) -> str:
        return f"{{{NS}}}{name}"

    scan_counter = 0
    for spectrum_elem in root.iter(tag("spectrum")):
        # Check MS level
        ms_level = 1
        for cv in spectrum_elem.iter(tag("cvParam")):
            if cv.attrib.get("name") == "ms level":
                ms_level = int(cv.attrib.get("value", 1))

        if ms_level != 2:
            continue

        scan_counter += 1

        # Extract precursor info
        precursor_mz = 0.0
        precursor_charge = 0
        rt = 0.0

        for scan_elem in spectrum_elem.iter(tag("scan")):
            for cv in scan_elem.iter(tag("cvParam")):
                if cv.attrib.get("name") == "scan start time":
                    try:
                        rt = float(cv.attrib.get("value", 0.0))
                        if cv.attrib.get("unitName", "") == "minute":
                            rt *= 60.0
                    except ValueError:
                        pass

        for prec_elem in spectrum_elem.iter(tag("selectedIon")):
            for cv in prec_elem.iter(tag("cvParam")):
                name = cv.attrib.get("name", "")
                val = cv.attrib.get("value", "0")
                if name == "selected ion m/z":
                    precursor_mz = float(val)
                elif name == "charge state":
                    precursor_charge = int(val)

        # Decode m/z and intensity arrays
        mz_arr = np.array([])
        int_arr = np.array([])

        arrays = list(spectrum_elem.iter(tag("binaryDataArray")))
        for bda in arrays:
            dtype_str = "64"
            compression = "none"
            array_type = "unknown"
            for cv in bda.iter(tag("cvParam")):
                name = cv.attrib.get("name", "")
                if "64" in name:
                    dtype_str = "64"
                elif "32" in name:
                    dtype_str = "32"
                if "zlib" in name:
                    compression = "zlib"
                if "m/z array" in name:
                    array_type = "mz"
                if "intensity array" in name:
                    array_type = "intensity"
            binary_elem = bda.find(tag("binary"))
            if binary_elem is not None and binary_elem.text:
                arr = decode_array(binary_elem.text.strip(), compression, dtype_str)
                if array_type == "mz":
                    mz_arr = arr
                elif array_type == "intensity":
                    int_arr = arr

        if len(mz_arr) > 0:
            yield MSMSSpectrum(
                scan_id=str(scan_counter),
                precursor_mz=precursor_mz,
                precursor_charge=precursor_charge,
                mz=mz_arr,
                intensity=int_arr if len(int_arr) == len(mz_arr) else np.ones_like(mz_arr),
                rt=rt,
            )


# ---------------------------------------------------------------------------
# Spectrum loading dispatcher
# ---------------------------------------------------------------------------

def load_spectra(filepath: str) -> list[MSMSSpectrum]:
    """
    Load all MS/MS spectra from a file, auto-detecting format by extension.

    Supported extensions: .mgf, .mzml, .mzML, .txt, .tsv
    """
    ext = Path(filepath).suffix.lower()
    if ext == ".mgf":
        return list(parse_mgf(filepath))
    elif ext in (".mzml",):
        return list(parse_mzml(filepath))
    elif ext in (".txt", ".tsv"):
        return [_load_plain_text(filepath)]
    else:
        raise ValueError(f"Unrecognised file extension: {ext}. "
                         f"Supported: .mgf, .mzml, .txt, .tsv")


def _load_plain_text(filepath: str) -> MSMSSpectrum:
    """Load a two-column (m/z, intensity) plain text spectrum."""
    data = np.loadtxt(filepath)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    mz = data[:, 0]
    intensity = data[:, 1] if data.shape[1] > 1 else np.ones(len(mz))
    return MSMSSpectrum(
        scan_id="1",
        precursor_mz=0.0,
        precursor_charge=0,
        mz=mz,
        intensity=intensity,
    )


# ---------------------------------------------------------------------------
# Spectrum preprocessing
# ---------------------------------------------------------------------------

class SpectrumPreprocessor:
    """
    Pipeline of preprocessing steps applied to raw MS/MS spectra before
    database search or de novo sequencing.

    Steps (all optional, controlled by constructor arguments):
      1. Noise filtering by absolute or relative intensity threshold.
      2. Peak picking — retain only the top-N most intense peaks.
      3. Deisotoping — remove isotope peaks within a window.
      4. Normalisation — scale intensities to base-peak = 1.

    Parameters
    ----------
    min_intensity_abs : float
        Remove peaks below this absolute intensity value. 0 = disabled.
    min_intensity_rel : float
        Remove peaks below this fraction of the base peak. 0 = disabled.
    top_n_peaks : int
        Retain only the N most intense peaks. 0 = retain all.
    deisotope : bool
        Whether to perform simple deisotoping.
    isotope_tolerance_da : float
        Mass tolerance (Da) for isotope peak detection.
    normalise : bool
        Whether to normalise intensities to base peak = 1.
    """

    PROTON = 1.007276
    NEUTRON = 1.003355   # average mass difference between isotope peaks

    def __init__(
        self,
        min_intensity_abs: float = 0.0,
        min_intensity_rel: float = 0.01,
        top_n_peaks: int = 200,
        deisotope: bool = True,
        isotope_tolerance_da: float = 0.02,
        normalise: bool = True,
    ) -> None:
        self.min_intensity_abs = min_intensity_abs
        self.min_intensity_rel = min_intensity_rel
        self.top_n_peaks = top_n_peaks
        self.deisotope = deisotope
        self.isotope_tolerance_da = isotope_tolerance_da
        self.normalise = normalise

    def process(self, spectrum: MSMSSpectrum) -> MSMSSpectrum:
        """Apply the full preprocessing pipeline to a spectrum."""
        mz = spectrum.mz.copy()
        intensity = spectrum.intensity.copy()

        if len(mz) == 0:
            return spectrum

        # Step 1: Noise filtering
        mask = np.ones(len(mz), dtype=bool)
        if self.min_intensity_abs > 0:
            mask &= intensity >= self.min_intensity_abs
        if self.min_intensity_rel > 0:
            mask &= intensity >= self.min_intensity_rel * intensity.max()
        mz, intensity = mz[mask], intensity[mask]

        # Step 2: Top-N peak selection
        if self.top_n_peaks > 0 and len(mz) > self.top_n_peaks:
            top_idx = np.argsort(intensity)[-self.top_n_peaks:]
            top_idx = np.sort(top_idx)
            mz, intensity = mz[top_idx], intensity[top_idx]

        # Step 3: Deisotoping
        if self.deisotope and len(mz) > 1:
            mz, intensity = self._deisotope(mz, intensity)

        # Step 4: Normalisation
        if self.normalise and len(intensity) > 0 and intensity.max() > 0:
            intensity = intensity / intensity.max()

        return MSMSSpectrum(
            scan_id=spectrum.scan_id,
            precursor_mz=spectrum.precursor_mz,
            precursor_charge=spectrum.precursor_charge,
            mz=mz,
            intensity=intensity,
            rt=spectrum.rt,
            title=spectrum.title,
        )

    def _deisotope(
        self, mz: np.ndarray, intensity: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Simple greedy deisotoping: for each peak, check whether a peak
        ~1.003 Da lower exists with higher intensity; if so, mark this
        peak as an isotope and remove it.
        """
        tol = self.isotope_tolerance_da
        n = len(mz)
        is_isotope = np.zeros(n, dtype=bool)

        for i in range(1, n):
            for charge in (1, 2):
                delta = self.NEUTRON / charge
                expected_mono = mz[i] - delta
                # Find candidate monoisotopic peak
                diffs = np.abs(mz[:i] - expected_mono)
                closest = np.argmin(diffs)
                if diffs[closest] <= tol and intensity[closest] >= intensity[i]:
                    is_isotope[i] = True
                    break

        return mz[~is_isotope], intensity[~is_isotope]

    def process_batch(self, spectra: list[MSMSSpectrum]) -> list[MSMSSpectrum]:
        """Apply preprocessing to a list of spectra."""
        return [self.process(s) for s in spectra]
