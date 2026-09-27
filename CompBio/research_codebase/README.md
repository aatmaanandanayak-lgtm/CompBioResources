# Computational Biology Research — Code Repository

This repository contains Python implementations from a few of my research projects in computational biology and biophysics.

---

## Repository Structure

```
research_codebase/
├── gnn_allostery/              #GNN-based allosteric site & pathway prediction
│   ├── data/                   #feature extraction and dataset construction
│   ├── models/                 #GNN architecture definitions
│   ├── training/               #training loop, loss, and evaluation
│   ├── inference/              #prediction and pathway scoring on new structures
│   └── utils/                  #graph construction and misc utilities
│
├── mitochondrial_etc/          #Spatial stochastic simulation of the ETC
│   ├── smoldyn_sim/            #smoldyn-based particle simulation interface (not the most successful)
│   ├── pde_sim/                #python stochastic spatial PDE implementation
│   └── analysis/               #bi-exponential fitting and parameter sweep analysis
│
├── rouse_polymer/              #Rouse model simulations for protein folding (validating SURJ article findings)
│   ├── simulation/             #rouse chain dynamics and confinement
│   └── analysis/               #rate extraction and model comparison
│
├── ms_peptide_fingerprinting/  #MS/MS peptide mass fingerprinting & sequencing
│   ├── io/                     #MGF/mzML spectrum parsing and preprocessing
│   ├── fragmentation/          #theoretical fragment ion generation (b/y/a/c/x/z)
│   ├── scoring/                #hyperscore, cosine similarity, de novo sequencing
│   └── identification/         #database search, FDR estimation, PSM reporting
│
└── electron_density_msd/       #Conformational flexibility from electron density
    ├── io/                     #CCP4/MRC map and PDB file parsing
    ├── analysis/               #MSD, density variance, RSCC, centroid displacement
    └── utils/                  #CLI entry point for flexibility analysis
```

---

## Projects

### 1. GNN-Based Allosteric Site and Pathway Prediction (`gnn_allostery/`)

Proteins are allosterically regulated when a binding event at one site propagates a functional change to a distal site. Predicting where these allosteric sites are is challenging enough (for a novel protein that hasn't been characterised) but understanding how signals could travel from an allosteric to an active site normally requires extensive spectroscopic characterisation (i.e., NMR CSP data, which itself is difficult to interpret if one isn't sure whether a CSP is due to direct binding with a ligand or the actual allosteric conformational change).

Proteins can surprisingly be represented very well using residue-level graphs: node features can encode amino acid (AA) identity, physicochemical properties, secondary structure propensity and solvent-accessible surface area (may be context dependent), B-factors, and evolutionary conservation scores (not the most exciting but can be useful), while edge features can capture chain distances, contact types (H-bonds, salt-bridges, hydrophobic contacts), and coevolutionary coupling strengths (potentially useful). This naturally led to the idea of developing a graph neural network (GNN) pipeline, with the model being trained on labeled allosteric sites from the Allosteric Database (ASD). A non-redundant sequence-based train/test split was used in the hopes of avoiding relationship memorisation from the training dataset. Equally, since most residues in proteins (even allosteric ones) do not actually play a role in propagation of the allosteric conformational change, focal loss was used to upweight minority allosteric residues. 

**Key dependencies:** PyTorch, PyTorch Geometric, BioPython, NumPy, scikit-learn

---

### 2. Spatial Stochastic Simulation of the Mitochondrial ETC (`mitochondrial_etc/`)

Respiratory ATP synthesis involves a series of redox reactions across complexes (I-IV) embedded in the mitochondrial membrane (forming the mitochondrial electron transport chain - ETC) and mediated by the mobile carriers ubiquinone and cytochrome c. While structural studies have shown that the large respiratory complexes can associate to form supercomplexes ('respirasomes'), the dependence of the kinetics of electron transfer on the formation of these species has been a topic of debate (a.k.a the solid-state vs the fluid-state model). The aim was to simulate redox processes within the ETC following the imposition of distinct constraints (to artificially drive the formation of a respirasome-like state) and look at the resulting electron flux and determine its dependence on each parameter.

This project implemented the ETC as a spatial stochastic simulation in two independent frameworks (that were not equally successful): a Smoldyn-interfacing particle-based simulation and a Python stochastic spatial PDE solver. Each respiratory complex was modelled as a state machine derived from a decomposition of its multi-step redox chemistry into tractable partial reactions that preserve physiological stoichiometry and kinetics. Parameter sweeps were run across key variables (complex densities, diffusion coefficients, rate constants), and carrier flux time courses were fitted to bi-exponential functions. Control runs and mathematical correction procedures were added to attempt to account for artefacts introduced by the simulation software.

**Key dependencies:** NumPy, SciPy, Matplotlib, pandas

---

### 3. Rouse Polymer Model Simulations for Protein Folding (`rouse_polymer/`)

Following the Stanford Undergraduate Research Journal's acceptance (for their Winter 2025 issue) of my statistical mechanical analysis of which of the two competing models of protein folding - the diffusion-collision model (DCM) vs the nucleation-condensation model (NCM) - would likely prevail in the crowded cellular milieu, I wanted to try and see if polymer simulations would add to my finding.

This project implemented Rouse chain simulations to model polypeptide dynamics under constraints including confinement geometries mimicking the GroEL/ES chaperonin cavity and effective crowding potentials. By then computing first-passage contact times between chain segments under DCM and NCM-like potentials, the simulations tested which model was consistent with the chaperone-mediated folding rate accelerations observed experimentally: under DCM-like potentials, the contact times were more consistent with experimental observations.

**Key dependencies:** NumPy, SciPy, Matplotlib

---

## Requirements

```bash
pip install torch torch-geometric biopython numpy scipy matplotlib pandas scikit-learn
```
---


### 4. MS/MS Peptide Mass Fingerprinting (`ms_peptide_fingerprinting/`)

Tandem mass spectrometry (MS/MS) generates fragment ion spectra that encode the amino acid sequence of a peptide. Identifying the originating peptide — either by database search or de novo sequencing — was the at the centre of one my projects to determine the nature of a disease marker in patient serum via shotgun proteomics.

This project implemented a complete peptide identification pipeline. The I/O module handles MGF and mzML spectral formats with a preprocessing pipeline covering noise filtering, deisotoping, and peak picking. The fragmentation module generates theoretical b/y/a/c/x/z ion series for any peptide sequence with arbitrary modifications (phosphorylation, oxidation, carbamidomethylation, etc.). Two scoring functions are implemented: a hyperscore (X!Tandem-style, rewarding both the number and intensity of matched ions) and a normalised cosine similarity between binned observed and theoretical spectra. A de novo sequencing module independently interpreted spectra as directed acyclic graph (DAG) paths through fragment mass differences. For database search, proteins were digested in silico with configurable proteases and peptide-spectrum matches were ranked with target-decoy false-discovery rate estimation.

**Key dependencies:** NumPy, SciPy

---

### 5. Conformational Flexibility from Electron Density Maps (`electron_density_msd/`)

Part of one my courseworks (in which I was investigating the conformational perturbation of a chaperone following ligand binding) required that I performed model-building of DnaK into an electron density map obtained using X-ray crystallography. Since the phenomenon of interest was conformational flexibility, I exploited the fact that regions of protein that are conformationally flexible or disordered manifest as diffuse density (which can be quantified independently of the crystallographic B-factor).

These scripts compute per-residue conformational flexibility from CCP4/MRC format electron density maps. Four complementary approaches were implemented: B-factor-derived MSD (MSD = B / 8π²); local density variance within a sphere around each Cα atom; the real-space correlation coefficient (RSCC) between the observed density and a Gaussian model density centred on the atom; and centroid displacement MSD across an ensemble of maps. A consensus flexibility score was then computed by normalising and combining the aforementioned metrics, and residues were ranked accordingly and output with a table and .csv file.

**Key dependencies:** NumPy, SciPy
