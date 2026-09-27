"""
electron_density_msd/io/density_io.py

Reading and parsing of electron density maps (CCP4/MRC format) and associated
atomic coordinate files (PDB/mmCIF) for conformational flexibility analysis.

CCP4/MRC format
---------------
The CCP4 map format stores a 3D grid of electron density values (typically
from X-ray crystallography or cryo-EM) as a binary file with a 1024-byte
header followed by the density data as 32-bit floats. The header encodes
grid dimensions, cell parameters (a, b, c, α, β, γ), axis ordering, and
origin information.

Usage
-----
    density = load_ccp4_map("2fo-fc.map")
    atoms = load_pdb_atoms("structure.pdb")
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


# Data containers

@dataclass
class ElectronDensityMap:
    """
    A 3D electron density grid read from a CCP4/MRC map file.

    Attributes
    ----------
    data : np.ndarray, shape (NX, NY, NZ)
        Density values on the grid. Axes correspond to the fast, medium,
        and slow axes as defined by the MAPC/MAPR/MAPS header fields,
        reordered to (X, Y, Z) during loading.
    cell_a, cell_b, cell_c : float
        Unit cell edge lengths in Ångströms.
    alpha, beta, gamma : float
        Unit cell angles in degrees.
    origin : np.ndarray, shape (3,)
        Grid origin in fractional or Ångström coordinates.
    voxel_size : np.ndarray, shape (3,)
        Size of a single voxel in Ångströms along (X, Y, Z).
    nx, ny, nz : int
        Grid dimensions along X, Y, Z.
    mean : float
        Mean density value of the map.
    std : float
        Standard deviation of density values.
    """
    data: np.ndarray
    cell_a: float
    cell_b: float
    cell_c: float
    alpha: float
    beta: float
    gamma: float
    origin: np.ndarray
    voxel_size: np.ndarray

    @property
    def nx(self) -> int:
        return self.data.shape[0]

    @property
    def ny(self) -> int:
        return self.data.shape[1]

    @property
    def nz(self) -> int:
        return self.data.shape[2]

    @property
    def mean(self) -> float:
        return float(self.data.mean())

    @property
    def std(self) -> float:
        return float(self.data.std())

    def fractional_to_cartesian(self, frac: np.ndarray) -> np.ndarray:
        """
        Convert fractional coordinates (u, v, w) to Cartesian (x, y, z) in Å.

        Uses the standard orthogonalisation matrix derived from cell parameters.
        """
        a, b, c = self.cell_a, self.cell_b, self.cell_c
        alpha_r = np.radians(self.alpha)
        beta_r = np.radians(self.beta)
        gamma_r = np.radians(self.gamma)

        cos_a = np.cos(alpha_r)
        cos_b = np.cos(beta_r)
        cos_g = np.cos(gamma_r)
        sin_g = np.sin(gamma_r)

        # Volume factor
        V = np.sqrt(
            1 - cos_a**2 - cos_b**2 - cos_g**2
            + 2 * cos_a * cos_b * cos_g
        )

        # Orthogonalisation matrix (CCP4 convention)
        M = np.array([
            [a, b * cos_g, c * cos_b],
            [0, b * sin_g, c * (cos_a - cos_b * cos_g) / sin_g],
            [0, 0,         c * V / sin_g],
        ])
        return M @ frac

    def grid_to_cartesian(self, grid_idx: np.ndarray) -> np.ndarray:
        """Convert grid indices (ix, iy, iz) to Cartesian coordinates (Å)."""
        frac = grid_idx / np.array([self.nx, self.ny, self.nz], dtype=float)
        return self.fractional_to_cartesian(frac)

    def interpolate_at(self, xyz: np.ndarray) -> float:
        """
        Trilinearly interpolate the density at a Cartesian point (Å).

        Parameters
        ----------
        xyz : np.ndarray (3,)
            Cartesian coordinates in Ångströms.

        Returns
        -------
        Interpolated density value (float).
        """
        # Convert to fractional
        frac = xyz / np.array([self.cell_a, self.cell_b, self.cell_c])
        # Convert to grid indices
        grid = frac * np.array([self.nx, self.ny, self.nz])

        # Clamp to grid bounds
        grid = np.clip(grid, 0, np.array([self.nx - 1, self.ny - 1, self.nz - 1]))

        i0 = np.floor(grid).astype(int)
        i1 = np.minimum(i0 + 1, np.array([self.nx - 1, self.ny - 1, self.nz - 1]))
        d = grid - i0   # interpolation weights

        # Trilinear interpolation
        c000 = self.data[i0[0], i0[1], i0[2]]
        c100 = self.data[i1[0], i0[1], i0[2]]
        c010 = self.data[i0[0], i1[1], i0[2]]
        c110 = self.data[i1[0], i1[1], i0[2]]
        c001 = self.data[i0[0], i0[1], i1[2]]
        c101 = self.data[i1[0], i0[1], i1[2]]
        c011 = self.data[i0[0], i1[1], i1[2]]
        c111 = self.data[i1[0], i1[1], i1[2]]

        c00 = c000 * (1 - d[0]) + c100 * d[0]
        c01 = c001 * (1 - d[0]) + c101 * d[0]
        c10 = c010 * (1 - d[0]) + c110 * d[0]
        c11 = c011 * (1 - d[0]) + c111 * d[0]

        c0 = c00 * (1 - d[1]) + c10 * d[1]
        c1 = c01 * (1 - d[1]) + c11 * d[1]

        return float(c0 * (1 - d[2]) + c1 * d[2])


# CCP4/MRC map reader

def load_ccp4_map(filepath: str) -> ElectronDensityMap:
    """
    Read a CCP4/MRC map file and return an ElectronDensityMap object.

    The CCP4 format has a 1024-byte (256 × 4-byte word) header followed by
    the density array. Key header words are:

      Words 1–3  : NX, NY, NZ — grid dimensions
      Words 4    : MODE (2 = 32-bit float, most common)
      Words 5–7  : NXSTART, NYSTART, NZSTART — grid start offsets
      Words 8–10 : MX, MY, MZ — sampling along unit cell edges
      Words 11–13: cell_a, cell_b, cell_c (Å)
      Words 14–16: alpha, beta, gamma (degrees)
      Words 17–19: MAPC, MAPR, MAPS — axis assignments (1=X, 2=Y, 3=Z)
      Words 20   : DMIN (minimum density)
      Words 21   : DMAX (maximum density)
      Words 25   : NSYMBT — number of bytes in symmetry records
      Words 50–52: ORIGIN (x, y, z in Å) — newer MRC extension
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Map file not found: {filepath}")

    with open(filepath, "rb") as fh:
        # Read 256 4-byte integers for the header
        header_bytes = fh.read(1024)
        header_ints = struct.unpack("256i", header_bytes)
        header_floats = struct.unpack("256f", header_bytes)

        # Detect endianness from MACHST (word 54) or by checking plausibility
        nx, ny, nz = header_ints[0], header_ints[1], header_ints[2]
        if not (0 < nx < 10000 and 0 < ny < 10000 and 0 < nz < 10000):
            # Try big-endian
            header_ints = struct.unpack(">256i", header_bytes)
            header_floats = struct.unpack(">256f", header_bytes)
            nx, ny, nz = header_ints[0], header_ints[1], header_ints[2]

        mode = header_ints[3]
        nxstart, nystart, nzstart = header_ints[4], header_ints[5], header_ints[6]
        mx, my, mz = header_ints[7], header_ints[8], header_ints[9]

        cell_a, cell_b, cell_c = header_floats[10], header_floats[11], header_floats[12]
        alpha, beta, gamma = header_floats[13], header_floats[14], header_floats[15]

        mapc, mapr, maps = header_ints[16], header_ints[17], header_ints[18]

        nsymbt = header_ints[23]   # bytes of symmetry records

        # MRC origin extension (words 50–52, 0-indexed = 49–51)
        origin = np.array([header_floats[49], header_floats[50], header_floats[51]])
        # Fallback: compute from NXSTART
        if np.allclose(origin, 0.0):
            origin = np.array([
                nxstart * cell_a / max(mx, 1),
                nystart * cell_b / max(my, 1),
                nzstart * cell_c / max(mz, 1),
            ])

        # Skip symmetry records
        fh.seek(1024 + nsymbt)

        # Read density data
        n_voxels = nx * ny * nz
        if mode == 2:
            dtype = np.float32
        elif mode == 1:
            dtype = np.int16
        elif mode == 0:
            dtype = np.int8
        elif mode == 6:
            dtype = np.uint16
        else:
            dtype = np.float32   # fallback

        raw = np.frombuffer(fh.read(n_voxels * np.dtype(dtype).itemsize), dtype=dtype)
        raw = raw.astype(np.float32)

    # Reshape to (NX, NY, NZ) — CCP4 stores slowest axis first: (NZ, NY, NX) or
    # per MAPC/MAPR/MAPS axis assignment
    data = raw.reshape((nz, ny, nx))

    # Reorder axes to (X, Y, Z) according to MAPC, MAPR, MAPS
    # MAPS = axis of slowest variation (first index in reshaped array)
    # MAPR = middle, MAPC = fastest
    axis_map = {maps: 0, mapr: 1, mapc: 2}   # current_axis → target_axis
    # Transpose to get (X=1, Y=2, Z=3) ordering
    perm = [axis_map.get(3, 2), axis_map.get(2, 1), axis_map.get(1, 0)]
    try:
        data = np.transpose(data, perm)
    except ValueError:
        pass  # if axis assignment is non-standard, keep as-is

    voxel_size = np.array([
        cell_a / max(mx, nx),
        cell_b / max(my, ny),
        cell_c / max(mz, nz),
    ])

    return ElectronDensityMap(
        data=data,
        cell_a=cell_a, cell_b=cell_b, cell_c=cell_c,
        alpha=alpha, beta=beta, gamma=gamma,
        origin=origin,
        voxel_size=voxel_size,
    )


# PDB atom reader

@dataclass
class Atom:
    """A single atom record from a PDB file."""
    serial: int
    name: str          # atom name (e.g. "CA", "N", "O")
    resname: str       # three-letter residue name
    chain: str         # chain identifier
    resseq: int        # residue sequence number
    x: float           # Cartesian x (Å)
    y: float           # Cartesian y (Å)
    z: float           # Cartesian z (Å)
    occupancy: float
    bfactor: float
    element: str

    @property
    def coords(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z])

    @property
    def is_backbone(self) -> bool:
        return self.name.strip() in ("CA", "N", "C", "O")

    @property
    def is_calpha(self) -> bool:
        return self.name.strip() == "CA"


def load_pdb_atoms(filepath: str, model: int = 1) -> list[Atom]:
    """
    Parse a PDB file and return a list of Atom objects.

    Reads ATOM and HETATM records. For multi-model files, only the specified
    model number is returned (default: first model).

    Parameters
    ----------
    filepath : str
        Path to .pdb file.
    model : int
        Model number to parse (1-indexed).
    """
    atoms: list[Atom] = []
    current_model = 1
    in_target_model = True

    with open(filepath, "r") as fh:
        for line in fh:
            record = line[:6].strip()

            if record == "MODEL":
                current_model = int(line[10:14].strip())
                in_target_model = (current_model == model)
                continue

            if record == "ENDMDL":
                if in_target_model:
                    break
                continue

            if record not in ("ATOM", "HETATM"):
                continue

            if not in_target_model:
                continue

            try:
                serial = int(line[6:11].strip())
                name = line[12:16].strip()
                resname = line[17:20].strip()
                chain = line[21].strip()
                resseq = int(line[22:26].strip())
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                occ = float(line[54:60].strip()) if len(line) > 60 else 1.0
                bfac = float(line[60:66].strip()) if len(line) > 66 else 0.0
                elem = line[76:78].strip() if len(line) > 78 else name[0]
            except (ValueError, IndexError):
                continue

            atoms.append(Atom(
                serial=serial,
                name=name,
                resname=resname,
                chain=chain,
                resseq=resseq,
                x=x, y=y, z=z,
                occupancy=occ,
                bfactor=bfac,
                element=elem,
            ))

    return atoms


def load_multiple_models(filepath: str) -> list[list[Atom]]:
    """
    Load all models from a multi-model PDB file (like an NMR ensemble).

    Returns a list of atom lists, one per model.
    """
    models: list[list[Atom]] = []
    current_atoms: list[Atom] = []
    in_model = False

    with open(filepath, "r") as fh:
        for line in fh:
            record = line[:6].strip()
            if record == "MODEL":
                in_model = True
                current_atoms = []
            elif record == "ENDMDL":
                if current_atoms:
                    models.append(current_atoms)
                current_atoms = []
                in_model = False
            elif record in ("ATOM", "HETATM") and in_model:
                try:
                    name = line[12:16].strip()
                    resname = line[17:20].strip()
                    chain = line[21].strip()
                    resseq = int(line[22:26].strip())
                    x, y, z = float(line[30:38]), float(line[38:46]), float(line[46:54])
                    occ = float(line[54:60]) if len(line) > 60 else 1.0
                    bfac = float(line[60:66]) if len(line) > 66 else 0.0
                    current_atoms.append(Atom(
                        serial=0, name=name, resname=resname,
                        chain=chain, resseq=resseq,
                        x=x, y=y, z=z,
                        occupancy=occ, bfactor=bfac, element=name[0],
                    ))
                except (ValueError, IndexError):
                    pass

    # Handle files without MODEL records
    if not models and current_atoms:
        models.append(current_atoms)

    return models
