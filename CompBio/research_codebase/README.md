# Computational Biology Research — Code Repository

This repository contains Python implementations from three independent research projects in computational biology and biophysics. Each project applies a distinct computational methodology to a specific biological problem.

---

## Repository Structure

```
research_codebase/
├── gnn_allostery/              # GNN-based allosteric site & pathway prediction
│   ├── data/                   # Feature extraction and dataset construction
│   ├── models/                 # GNN architecture definitions
│   ├── training/               # Training loop, loss, and evaluation
│   ├── inference/              # Prediction and pathway scoring on new structures
│   └── utils/                  # Graph construction and misc utilities
│
├── mitochondrial_etc/          # Spatial stochastic simulation of the ETC
│   ├── smoldyn_sim/            # Smoldyn-based particle simulation interface
│   ├── pde_sim/                # Python stochastic spatial PDE implementation
│   └── analysis/               # Bi-exponential fitting and parameter sweep analysis
│
├── rouse_polymer/              # Rouse model simulations for protein folding
│   ├── simulation/             # Rouse chain dynamics and confinement
│   └── analysis/               # Rate extraction and model comparison
│
├── ms_peptide_fingerprinting/  # MS/MS peptide mass fingerprinting & sequencing
│   ├── io/                     # MGF/mzML spectrum parsing and preprocessing
│   ├── fragmentation/          # Theoretical fragment ion generation (b/y/a/c/x/z)
│   ├── scoring/                # Hyperscore, cosine similarity, de novo sequencing
│   └── identification/         # Database search, FDR estimation, PSM reporting
│
└── electron_density_msd/       # Conformational flexibility from electron density
    ├── io/                     # CCP4/MRC map and PDB file parsing
    ├── analysis/               # MSD, density variance, RSCC, centroid displacement
    └── utils/                  # CLI entry point for flexibility analysis
```

---

## Projects

### 1. GNN-Based Allosteric Site and Pathway Prediction (`gnn_allostery/`)

Proteins are allosterically regulated when a binding event at one site propagates a functional change to a distal site. Predicting where these allosteric sites are, and how signals travel between them, is a central challenge in drug discovery.

This project builds a graph neural network pipeline in which proteins are represented as residue-level graphs. Node features encode amino acid identity, physicochemical properties, secondary structure, solvent-accessible surface area, B-factors, and evolutionary conservation scores. Edge features capture inter-residue distances, contact types (hydrogen bonds, salt bridges, hydrophobic contacts), and coevolutionary coupling strengths. The model is trained on labeled allosteric sites from the Allosteric Database (ASD), with non-redundant sequence-based train/test splits and focal loss to address class imbalance. A second model head outputs per-edge importance scores to predict allosteric communication pathways between predicted allosteric pockets and known active sites.

**Key dependencies:** PyTorch, PyTorch Geometric, BioPython, NumPy, scikit-learn

---

### 2. Spatial Stochastic Simulation of the Mitochondrial ETC (`mitochondrial_etc/`)

The mitochondrial electron transport chain (ETC) drives ATP synthesis via a series of redox reactions across Complexes I–IV, with mobile carriers (ubiquinone, cytochrome c) shuttling electrons between complexes. Understanding how changes in individual biophysical parameters affect overall electron flux is critical for studying mitochondrial disease and aging.

This project implements the ETC as a spatial stochastic simulation in two independent frameworks: a Smoldyn-interfacing particle-based simulation, and a Python stochastic spatial PDE solver. Each respiratory complex is modelled as a state machine derived from a decomposition of its multi-step redox chemistry into tractable partial reactions that preserve physiological stoichiometry and kinetics. Parameter sweeps are run across key variables (complex densities, diffusion coefficients, rate constants), and carrier flux time courses are fitted to bi-exponential functions. Control runs and mathematical correction procedures account for artefacts introduced by the simulation software.

**Key dependencies:** NumPy, SciPy, Matplotlib, pandas

---

### 3. Rouse Polymer Model Simulations for Protein Folding (`rouse_polymer/`)

Two competing models describe how proteins fold: the diffusion-collision model (DCM), in which independently folded microdomains diffuse and coalesce, and the nucleation-condensation model (NCM), in which structure forms cooperatively around a nucleus. Discriminating between them in the cellular environment — where macromolecular crowding and chaperone confinement impose additional constraints — requires going beyond in vitro kinetics.

This project implements Rouse chain simulations to model polypeptide dynamics under physiologically relevant constraints, including confinement geometries mimicking the GroEL/ES chaperonin cavity and effective crowding potentials. By computing first-passage contact times between chain segments under DCM and NCM-like potentials, the simulations test which model is consistent with the chaperone-mediated folding rate accelerations observed experimentally. The results support the prevalence of the DCM in vivo and were published in the Winter 2025 issue of the *Stanford Undergraduate Research Journal*.

**Key dependencies:** NumPy, SciPy, Matplotlib

---

## Requirements

```bash
pip install torch torch-geometric biopython numpy scipy matplotlib pandas scikit-learn
```

---

## Citation

If any part of this code is useful to your work, please cite:

> "Prevalence of the Diffusion Collision Model of Protein Folding In Vivo", *Stanford Undergraduate Research Journal*, Winter 2025.

---

### 4. MS/MS Peptide Mass Fingerprinting (`ms_peptide_fingerprinting/`)

Tandem mass spectrometry (MS/MS) generates fragment ion spectra that encode the amino acid sequence of a peptide. Identifying the originating peptide — either by database search or de novo sequencing — is the central problem in shotgun proteomics.

This project implements a complete peptide identification pipeline. The I/O module handles MGF and mzML spectral formats with a preprocessing pipeline covering noise filtering, deisotoping, and peak picking. The fragmentation module generates theoretical b/y/a/c/x/z ion series for any peptide sequence with arbitrary modifications (phosphorylation, oxidation, carbamidomethylation, etc.). Two scoring functions are implemented: a hyperscore (X!Tandem-style, rewarding both the number and intensity of matched ions) and a normalised cosine similarity between binned observed and theoretical spectra. A de novo sequencing module independently interprets spectra as directed acyclic graph (DAG) paths through fragment mass differences. For database search, proteins are digested in silico with configurable proteases and PSMs are ranked with target-decoy FDR estimation.

**Key dependencies:** NumPy, SciPy

---

### 5. Conformational Flexibility from Electron Density Maps (`electron_density_msd/`)

In X-ray crystallography, the electron density map directly encodes atomic positional information. Regions of the protein that are conformationally flexible or disordered manifest as broad, weak, or diffuse density, which can be quantified independently of the crystallographic B-factor.

This project computes per-residue conformational flexibility from CCP4/MRC format electron density maps. Four complementary methods are implemented: B-factor-derived MSD (MSD = B / 8π²); local density variance within a sphere around each Cα atom; the real-space correlation coefficient (RSCC) between the observed density and a Gaussian model density centred on the atom; and centroid displacement MSD across an ensemble of maps. A consensus flexibility score is computed by normalising and combining all enabled metrics, and residues are ranked accordingly. Output is a formatted table and a CSV file.

**Key dependencies:** NumPy, SciPy
