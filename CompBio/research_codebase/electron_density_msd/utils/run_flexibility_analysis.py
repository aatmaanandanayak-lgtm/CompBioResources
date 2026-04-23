"""
electron_density_msd/utils/run_flexibility_analysis.py

Command-line entry point for computing per-residue conformational flexibility
from electron density maps and atomic coordinate files.

Usage
-----
    python run_flexibility_analysis.py \
        --map 2fofc.map \
        --pdb structure.pdb \
        --output flexibility.csv \
        --top_n 25 \
        --radius 2.5 \
        --sigma 0.8

For ensemble/multi-map analysis (e.g. time-resolved crystallography):
    python run_flexibility_analysis.py \
        --map map_1.map map_2.map map_3.map \
        --pdb structure.pdb \
        --ensemble \
        --output flexibility_ensemble.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from electron_density_msd.io.density_io import load_ccp4_map, load_pdb_atoms
from electron_density_msd.analysis.msd_from_density import (
    ResidueFlexibilityAnalyser,
    print_flexibility_report,
    save_flexibility_csv,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute per-residue conformational flexibility from electron density maps."
    )
    parser.add_argument(
        "--map", nargs="+", required=True,
        help="Path(s) to CCP4/MRC map file(s). Provide multiple maps for ensemble analysis.",
    )
    parser.add_argument(
        "--pdb", required=True,
        help="Path to PDB coordinate file.",
    )
    parser.add_argument(
        "--output", default="flexibility.csv",
        help="Output CSV file path.",
    )
    parser.add_argument(
        "--top_n", type=int, default=20,
        help="Number of most flexible residues to display.",
    )
    parser.add_argument(
        "--radius", type=float, default=2.5,
        help="Sphere radius (Å) for local density extraction.",
    )
    parser.add_argument(
        "--sigma", type=float, default=0.8,
        help="Gaussian width (Å) for RSCC model density.",
    )
    parser.add_argument(
        "--ensemble", action="store_true",
        help="Treat multiple maps as an ensemble for centroid MSD computation.",
    )
    parser.add_argument(
        "--no_bfactor", action="store_true",
        help="Exclude B-factor MSD from consensus score.",
    )
    parser.add_argument(
        "--no_variance", action="store_true",
        help="Exclude density variance from consensus score.",
    )
    parser.add_argument(
        "--no_rscc", action="store_true",
        help="Exclude RSCC from consensus score.",
    )
    args = parser.parse_args()

    # Load primary map (first in list)
    print(f"Loading primary density map: {args.map[0]}")
    primary_map = load_ccp4_map(args.map[0])
    print(f"  Grid: {primary_map.nx} × {primary_map.ny} × {primary_map.nz}")
    print(f"  Cell: {primary_map.cell_a:.2f} × {primary_map.cell_b:.2f} × {primary_map.cell_c:.2f} Å")
    print(f"  Density: mean={primary_map.mean:.4f}, σ={primary_map.std:.4f}")

    # Load ensemble maps (if requested)
    ensemble_maps = []
    if args.ensemble and len(args.map) > 1:
        for map_path in args.map[1:]:
            print(f"Loading ensemble map: {map_path}")
            ensemble_maps.append(load_ccp4_map(map_path))
        print(f"  {len(ensemble_maps)} additional maps loaded for ensemble analysis.")

    # Load atomic model
    print(f"\nLoading PDB coordinates: {args.pdb}")
    atoms = load_pdb_atoms(args.pdb)
    calpha_count = sum(1 for a in atoms if a.is_calpha)
    print(f"  {len(atoms)} atoms loaded ({calpha_count} Cα atoms).")

    # Run analysis
    print()
    analyser = ResidueFlexibilityAnalyser(
        density_map=primary_map,
        atoms=atoms,
        ensemble_maps=ensemble_maps if args.ensemble else [],
        sphere_radius=args.radius,
        gaussian_sigma=args.sigma,
        use_bfactor=not args.no_bfactor,
        use_density_variance=not args.no_variance,
        use_rscc=not args.no_rscc,
        use_centroid_msd=args.ensemble and len(ensemble_maps) > 1,
    )
    results = analyser.analyse()

    # Report
    print_flexibility_report(results, top_n=args.top_n)

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    save_flexibility_csv(results, args.output)


if __name__ == "__main__":
    main()
