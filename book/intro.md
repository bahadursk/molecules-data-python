# Coding Molecules: Python, Data Science and AI for Chemistry and Materials Science

```{image} _static/library_table.png
:alt: "Periodic Table of Python for Chemists: coloured tiles for every library and method used in the book, such as NumPy, pandas, Matplotlib, RDKit, SciPy, scikit-learn and PyTorch, each labelled with the chapter that introduces it"
:width: 100%
```

## An interactive introduction to Python, cheminformatics and machine learning for chemists

*Coding Molecules: Python, Data Science and AI for Chemistry and Materials Science* teaches programming the way chemists actually use it: to calculate, to plot, to draw and search molecules, to simulate, and to learn from data. It starts at the very beginning (your first line of Python is a molar mass) and ends at the research frontier, with machine learning for drug discovery, catalysis, spectroscopy, batteries, polymers and materials, and with AI-driven discovery, sensing, imaging and prediction.

Every chapter is a Jupyter notebook. You can read it here, or press the 🚀 **launch** button at the top of any chapter to open it in **Google Colab** or **Binder** and run, edit and break the code yourself, with nothing to install. The ⬇️ **download** button gives you the notebook to run on your own computer.

Python was chosen because it is:

- **easy to start** and readable, even for people who have never programmed;
- **free and open source**, and it runs on Windows, macOS and Linux;
- **the language of modern chemical data science**, with mature libraries for numerics (NumPy, SciPy), data (pandas), plotting (Matplotlib, Seaborn, Plotly), molecules (RDKit), quantum chemistry (PySCF), materials (ASE, pymatgen) and machine learning (scikit-learn, PyTorch);
- **backed by a large, welcoming scientific community.**

## Who this book is for

| Reader | Start at | Typical path |
|---|---|---|
| School and first-year students | Chapter 1 | Parts I–II, then the 🟢 sections of Parts III–IV |
| Undergraduates and lab scientists | Chapter 12 (skim Part I) | Parts II–IV, then Part V |
| Graduate students and researchers | Chapter 31 (dip into II–IV as needed) | Parts V–VII |
| Teachers | the chapter guide below | each chapter is self-contained, with graded exercises |

No background beyond school chemistry is assumed for Parts I–III. Every chapter carries a level badge:

```{admonition} 🟢 Beginner · school & first-year
:class: level-1
No programming or advanced chemistry assumed.
```
```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Part I and general chemistry.
```
```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Research-grade methods and current literature.
```

## What makes this book different

- **Chemistry first.** Every Python idea is introduced through a chemical problem: molar masses, titrations, spectra, kinetics, crystal structures.
- **Molecules everywhere.** Structures, reaction schemes and colour-highlighted functional groups drawn live with RDKit and py3Dmol.
- **Machine learning across all of chemistry.** Three parts take ML from first principles into medicinal, organic, analytical, physical and quantum chemistry, catalysis, polymers, energy materials and materials discovery, and on to current research.
- **Honest science.** Scaffold and time-based splits, uncertainty, applicability domains and reproducibility are taught alongside every model.
- **Free to reuse.** All text, code, figures and data are original and openly licensed.

## Organisation of the book

The first two parts are the foundation: Part I teaches the Python language and Part II the scientific libraries that everything else uses. After that, the parts are largely independent, so you can follow your interests. Each chapter ends with **Further Reading** (the official documentation of the libraries it uses) and graded **Exercises** (★ to ★★★), with worked solutions in drop-downs and in Appendix D.

| Ch. | Title | Level | What you will learn |
|---|---|:-:|---|
| | **Part I · Python Fundamentals** | | |
| 1 | [Python as Your Chemistry Calculator](part1_python/01_python_calculator.ipynb) | 🟢 | Run code in Jupyter, Colab and VS Code; Arithmetic, operator precedence and scientific notation; Variables and naming |
| 2 | [Numbers, Booleans and Types](part1_python/02_data_types.ipynb) | 🟢 | int, float, complex and bool; Type conversion; Floating-point precision and why 0.1 + 0.2 != 0.3 |
| 3 | [Strings: Formulas, Names and SMILES](part1_python/03_strings.ipynb) | 🟢 | Indexing and slicing; String methods; Parsing a chemical formula by hand |
| 4 | [Lists, Tuples, Sets and Dictionaries](part1_python/04_collections.ipynb) | 🟢 | Lists of elements and masses; Tuples for immutable data (coordinates); Sets for unique ions |
| 5 | [Decisions and Loops](part1_python/05_control_flow.ipynb) | 🟢 | if / elif / else (acid, base or neutral?); for and while loops; range, enumerate and zip |
| 6 | [Writing Your Own Functions](part1_python/06_functions.ipynb) | 🟢 | def, arguments and return values; Default and keyword arguments; Docstrings and type hints |
| 7 | [Pythonic Code: Comprehensions, Generators and Idioms](part1_python/07_comprehensions.ipynb) | 🟡 | List, dict and set comprehensions; Generators and lazy evaluation; Unpacking |
| 8 | [Reading and Writing Chemical Files](part1_python/08_files.ipynb) | 🟡 | Paths with pathlib; Text files and the with statement; CSV and JSON |
| 9 | [Errors, Exceptions and Debugging](part1_python/09_errors.ipynb) | 🟡 | Reading a traceback; try / except / finally; Raising meaningful errors |
| 10 | [Object-Oriented Chemistry: Atoms and Molecules as Classes](part1_python/10_oop.ipynb) | 🟡 | Classes and instances; Attributes and methods; Special methods (__repr__, __add__) |
| 11 | [Modules, Packages and Environments](part1_python/11_modules_envs.ipynb) | 🟡 | import and the standard library; pip and conda; Virtual environments |
| | **Part II · The Scientific Python Stack** | | |
| 12 | [NumPy: Arrays for Chemical Data](part2_scipy_stack/12_numpy.ipynb) | 🟡 | Creating arrays; Vectorised maths and broadcasting; Indexing, slicing and masks |
| 13 | [Matplotlib: Publication-Quality Figures](part2_scipy_stack/13_matplotlib.ipynb) | 🟢 | Line, scatter and bar plots; Spectra and titration curves; Subplots and layouts |
| 14 | [Seaborn and Plotly: Statistical and Interactive Graphics](part2_scipy_stack/14_seaborn_plotly.ipynb) | 🟡 | Distributions, pair plots and heatmaps; Interactive spectra with Plotly; 3D surfaces (potential energy surfaces) |
| 15 | [pandas: Tables of Molecules and Measurements](part2_scipy_stack/15_pandas.ipynb) | 🟡 | Series and DataFrames; Reading CSV/Excel lab data; Selecting, filtering and sorting |
| 16 | [SciPy: Fitting, Integration and Optimisation](part2_scipy_stack/16_scipy.ipynb) | 🟡 | Curve fitting (Beer–Lambert, Michaelis–Menten); Numerical integration (peak areas); Root finding (equilibrium) |
| 17 | [SymPy: Symbolic Chemistry](part2_scipy_stack/17_sympy.ipynb) | 🟡 | Symbols and expressions; Solving equations (ICE tables); Calculus (rate laws) |
| 18 | [Units and Uncertainty: pint and uncertainties](part2_scipy_stack/18_units_uncertainty.ipynb) | 🟡 | Unit-aware calculations; Unit conversion; Error propagation |
| | **Part III · Cheminformatics and Molecular Visualisation** | | |
| 19 | [Representing Molecules: SMILES, InChI and Molfiles](part3_cheminformatics/19_representing_molecules.ipynb) | 🟡 | Molecular graphs; SMILES grammar; Canonicalisation |
| 20 | [RDKit I: Drawing and Describing Molecules](part3_cheminformatics/20_rdkit_basics.ipynb) | 🟢 | Molecules from SMILES; Drawing and grids; Colour highlighting |
| 21 | [RDKit II: Substructures, SMARTS and Fingerprints](part3_cheminformatics/21_substructure_similarity.ipynb) | 🟡 | Substructure search; SMARTS patterns for functional groups; Morgan and MACCS fingerprints |
| 22 | [RDKit III: Chemical Reactions in Code](part3_cheminformatics/22_reactions.ipynb) | 🔴 | Reaction SMARTS; Running reactions; Drawing reaction schemes |
| 23 | [3D Structures, Conformers and py3Dmol](part3_cheminformatics/23_3d_structures.ipynb) | 🟡 | Embedding 3D coordinates; Force-field optimisation (MMFF, UFF); Conformer ensembles |
| 24 | [Chemical Databases from Python](part3_cheminformatics/24_databases.ipynb) | 🟡 | PubChem with PubChemPy; ChEMBL web services; RCSB PDB |
| | **Part IV · Chemistry Applications** | | |
| 25 | [Chemical Kinetics and ODEs](part4_applications/25_kinetics.ipynb) | 🟡 | Rate laws; solve_ivp; Consecutive and parallel reactions |
| 26 | [Thermodynamics and Equilibrium](part4_applications/26_thermo_equilibrium.ipynb) | 🟡 | ΔG, ΔH, ΔS calculations; Equilibrium constants; Speciation diagrams (pH) |
| 27 | [Spectroscopy and Signal Processing](part4_applications/27_spectroscopy.ipynb) | 🟡 | UV–Vis and Beer–Lambert; IR peak picking; NMR: chemical shifts, multiplets, FFT |
| 28 | [Quantum Chemistry with Python](part4_applications/28_quantum.ipynb) | 🔴 | Particle in a box and harmonic oscillator; Hydrogen orbitals; Hückel MO theory |
| 29 | [Molecular Simulation and MD Analysis](part4_applications/29_molecular_dynamics.ipynb) | 🔴 | Lennard-Jones and Monte Carlo; A toy MD engine; Trajectory analysis with MDAnalysis |
| 30 | [Crystals and Materials: ASE and pymatgen](part4_applications/30_materials.ipynb) | 🔴 | Unit cells and lattices; Building crystals; Powder XRD patterns |
| | **Part V · Machine Learning Foundations for Chemists** | | |
| 31 | [Machine Learning with scikit-learn](part5_ml_foundations/31_ml_intro.ipynb) | 🟡 | The ML workflow; Train / test splits and cross-validation; Linear and kernel regression |
| 32 | [Featurising Molecules and Materials](part5_ml_foundations/32_featurisation.ipynb) | 🟡 | Descriptors and fingerprints; Graphs and SMILES tokens; Composition features (matminer) |
| 33 | [Mapping Chemical Space](part5_ml_foundations/33_chemical_space.ipynb) | 🟡 | PCA, t-SNE and UMAP; Clustering (Butina, k-means); Scaffold analysis |
| 34 | [Deep Learning with PyTorch](part5_ml_foundations/34_deep_learning.ipynb) | 🔴 | Tensors and autograd; Feed-forward networks; Graph neural networks |
| 35 | [Bayesian Optimisation and Active Learning](part5_ml_foundations/35_bayesian_opt.ipynb) | 🔴 | Gaussian processes; Acquisition functions; Closed-loop and self-driving labs |
| | **Part VI · Machine Learning Across Chemistry and Materials Science** | | |
| 36 | [Medicinal Chemistry: QSAR, ADMET and Virtual Screening](part6_ml_applications/36_ml_drug_discovery.ipynb) | 🔴 | QSAR/QSPR models; Solubility (ESOL) and ADMET prediction; Docking-score surrogates |
| 37 | [Organic Chemistry: Reaction Outcome, Yield and Retrosynthesis](part6_ml_applications/37_ml_organic_synthesis.ipynb) | 🔴 | Reaction fingerprints; Yield prediction (Buchwald–Hartwig HTE data); Condition recommendation |
| 38 | [Catalysis: Descriptors, Volcano Plots and Screening](part6_ml_applications/38_ml_catalysis.ipynb) | 🔴 | Scaling relations and d-band descriptors; Volcano plots; Predicting adsorption energies |
| 39 | [Analytical Chemistry: ML for Spectra](part6_ml_applications/39_ml_spectroscopy.ipynb) | 🔴 | Chemometrics: PLS and PCA; IR/Raman classification; NMR shift prediction |
| 40 | [Physical and Quantum Chemistry: ML Potentials](part6_ml_applications/40_ml_physical_chem.ipynb) | 🔴 | Learning energies from QM9; Δ-learning; Machine-learned interatomic potentials |
| 41 | [Materials Science: Property Prediction and Discovery](part6_ml_applications/41_ml_materials.ipynb) | 🔴 | Band gaps and formation energies; Crystal graph networks; Stability and the convex hull |
| 42 | [Energy Materials: Batteries, Photovoltaics and MOFs](part6_ml_applications/42_ml_energy_materials.ipynb) | 🔴 | Battery electrolytes and electrodes; Perovskite solar cells; MOF gas adsorption |
| 43 | [Polymers and Soft Matter](part6_ml_applications/43_ml_polymers.ipynb) | 🔴 | Representing polymers (BigSMILES); Glass-transition prediction; Copolymer design |
| 44 | [Biochemistry and Environmental Chemistry](part6_ml_applications/44_ml_bio_env.ipynb) | 🔴 | Protein–ligand binding; Enzyme function; Toxicity and environmental fate |
| 45 | [Generative Models and Inverse Design](part6_ml_applications/45_generative.ipynb) | 🔴 | VAEs and SMILES generation; Graph and diffusion models; Reinforcement learning for design |
| 46 | [Large Language Models and AI Agents in Chemistry](part6_ml_applications/46_llms_chemistry.ipynb) | 🔴 | Chemistry LLMs; Text mining the literature; Tool-using agents |
| | **Part VII · Research Frontiers: AI-Assisted Chemistry** | | |
| 47 | [AI-Assisted Discovery of Molecules and Materials](part7_frontiers/47_ai_discovery.ipynb) | 🔴 | The discovery loop: predict, select, make, measure; Virtual screening at scale; Multi-objective optimisation and Pareto fronts |
| 48 | [AI for Chemical Sensing: Electronic Noses and Tongues](part7_frontiers/48_ai_sensing.ipynb) | 🔴 | Sensor arrays and pattern recognition; Drift correction; Gas identification and quantification |
| 49 | [AI for Chemical and Microscopy Imaging](part7_frontiers/49_ai_imaging.ipynb) | 🔴 | Images as arrays; Particle segmentation in electron micrographs; Hyperspectral and Raman imaging |
| 50 | [Predictive Chemistry: Reactions, Processes and Digital Twins](part7_frontiers/50_ai_prediction.ipynb) | 🔴 | Kinetic models plus ML (hybrid modelling); Process monitoring and soft sensors; Forecasting with uncertainty |
| 51 | [Trustworthy, Explainable and Open AI for Science](part7_frontiers/51_ai_trust.ipynb) | 🔴 | Explainability; Uncertainty and out-of-distribution detection; FAIR data and open science |

| Appendix | Contents |
|---|---|
| [A](appendices/A1_cheatsheets.md) | Python, NumPy, pandas, plotting, RDKit, scikit-learn and PyTorch cheat sheets |
| [B](appendices/A2_datasets.md) | Every dataset used in the book, and how it was generated |
| [C](appendices/A3_glossary.md) | Glossary of chemistry, programming and ML terms |
| [D](appendices/A4_solutions.md) | Worked solutions to selected exercises |
| [E](appendices/A5_references.md) | Full reference list |

## Licence

This book is © 2026 Dr. Bahadur Sk. The text, figures and data are released under the [Creative Commons Attribution 4.0 International licence (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), and the code under the [MIT licence](https://opensource.org/license/mit). You may copy, adapt and share the material for any purpose, including teaching, provided you give appropriate credit. All datasets are generated by the book's own scripts, and published research is referred to by citation only.

## How to cite this book

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24583664.svg)](https://doi.org/10.5281/zenodo.24583664)

> Sk, B. (2026). *Coding Molecules: Python, Data Science and AI for Chemistry and Materials Science* [Interactive book]. Zenodo. <https://doi.org/10.5281/zenodo.24583664>

This DOI covers all versions and always resolves to the latest release. To cite version 1.0 specifically, use [10.5281/zenodo.24583665](https://doi.org/10.5281/zenodo.24583665).

```bibtex
@book{sk2026molecules,
  author    = {Sk, Bahadur},
  title     = {Coding Molecules: Python, Data Science and AI for Chemistry and Materials Science},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.24583664},
  url       = {https://doi.org/10.5281/zenodo.24583664}
}
```
