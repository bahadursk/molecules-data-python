# Datasets Used in This Book

Every dataset in this book is either generated inside the notebook that uses it or produced by one script, [`tools/make_datasets.py`](https://github.com/bahadursk/molecules-data-python/blob/main/tools/make_datasets.py). The book therefore carries **no third-party data and no licensing restrictions**: you may copy, modify and redistribute every file under the book's CC-BY-4.0 licence.

```{admonition} Read this before using any number in these files
:class: warning
The **chemical structures** are real: they are built by combining common scaffolds and substituents, and every SMILES string parses in RDKit. The **property values** are *simulated* from transparent, physically motivated formulas plus random noise. Every simulated column ends in `_sim` (or lives in a file whose name does). They behave like real data for teaching, with realistic correlations, noise and outliers, but **they are not measurements and must never be quoted as experimental values.**
```

To regenerate all four files from scratch:

```bash
python tools/make_datasets.py
```

The script uses a fixed random seed (2026), so the files are reproducible byte for byte with the same library versions.

## How notebooks load the data

Each chapter tries the local copy first and falls back to the copy on GitHub, so the notebooks work in Colab and Binder without any setup:

```python
DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"

def load_data(name, **kw):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name, **kw)
```

## `molecules_sim.csv` — 2 660 small organic molecules

Used in Chapters 31, 32, 33, 34, 36, 44, 45, 46, 47 and 51.

| Column | Meaning | Origin |
|---|---|---|
| `smiles` | canonical SMILES | built from scaffold × substituent combinations |
| `scaffold` | Bemis–Murcko scaffold SMILES (empty for acyclic molecules) | RDKit |
| `MW` | molecular weight / g mol⁻¹ | RDKit `Descriptors.MolWt` |
| `logP` | Crippen logP | RDKit `Crippen.MolLogP` |
| `TPSA` | topological polar surface area / Å² | RDKit |
| `HBD` | hydrogen-bond donors | RDKit |
| `RotB` | rotatable bonds | RDKit |
| `logS_sim` | simulated aqueous solubility, log₁₀(mol L⁻¹) | ESOL-style linear relation {cite:p}`delaney2004` with an added H-bonding term and Gaussian noise (σ = 0.35) |
| `pIC50_sim` | simulated potency against a *fictitious* kinase | rewards an H-bond donor next to an aromatic N (a "hinge-binder" motif), an aryl halogen, moderate logP and MW ≈ 320; noise σ = 0.4 |
| `toxic_sim` | simulated binary toxicity flag | logistic function of the number of structural alerts and logP, then sampled |
| `alerts` | which structural alerts matched (`;`-separated) | SMARTS patterns in the script |

```{admonition} ⚗️ Chemistry first
:class: chem-note
Because the descriptors are computed from the real structures, the *relationships* between them (MW versus logP, TPSA versus HBD) are genuine chemistry. Only the three `_sim` targets are invented. This is why models trained on this file behave realistically: they face real structural correlations and real scaffold imbalance (benzene and acyclic molecules dominate), which is exactly what makes scaffold splits matter.
```

## `cn_coupling_hte_sim.csv` — 960 simulated C–N coupling reactions

Used in Chapters 35, 37 and 38. A full-factorial "high-throughput experimentation" plate in the style of published Buchwald–Hartwig screens {cite:p}`ahneman2018`, with simulated yields.

| Column | Meaning |
|---|---|
| `aryl_halide`, `aryl_halide_smiles` | 8 aryl halides (Cl, Br, I; electron-rich, electron-poor and heteroaryl) |
| `ligand`, `ligand_electron_richness`, `ligand_cone_angle` | 4 fictitious phosphine ligands described by two numbers |
| `base`, `base_pKa` | 5 common bases with approximate conjugate-acid pKa |
| `additive` | 6 additives, two of which poison the catalyst |
| `yield_sim` | simulated yield / %, a logistic function of halide reactivity (I > Br > Cl), ligand electronics, a cone-angle optimum near 185°, base strength and additive effects, plus noise (σ = 4 %), clipped to 0–100 |

8 × 4 × 5 × 6 = 960 rows. The yield surface has a genuine interaction (electron-rich ligands help aryl chlorides most), so linear models underfit and tree ensembles do well.

## `binary_materials_sim.csv` — 770 binary compositions

Used in Chapter 41. Every combination of 14 electropositive elements (A) and 11 electronegative elements (B) in five stoichiometries.

| Column | Meaning | Origin |
|---|---|---|
| `formula` | e.g. `NaCl`, `Al2O3` | constructed |
| `A`, `B`, `nA`, `nB` | elements and their counts | constructed |
| `chi_A`, `chi_B`, `r_A`, `r_B`, `Z_A`, `Z_B` | Pauling electronegativity, atomic radius / Å, atomic number | pymatgen element data {cite:p}`ong2013` |
| `charge_balanced` | 1 if the common oxidation states sum to zero | computed |
| `band_gap_sim_eV` | simulated band gap / eV | grows with electronegativity difference, shrinks with atomic size, near zero if not charge-balanced |
| `formation_energy_sim_eV_atom` | simulated formation energy / eV atom⁻¹ | more negative with larger electronegativity difference when charge-balanced |

```{admonition} A trap worth knowing
:class: warning
The formula **`NaN`** (sodium nitride, Na + N) is read by pandas as a missing value. Load this file with `keep_default_na=False, na_values=[""]`, as Chapter 41 does.
```

## `polymers_sim.csv` — 48 polymers and glass-transition temperatures

Used in Chapter 43.

| Column | Meaning |
|---|---|
| `name` | polymer name |
| `repeat_unit` | repeat-unit SMILES with `*` marking the connection points |
| `Tg_C` | glass-transition temperature / °C |
| `source` | either `approximate textbook value` or `simulated` |

Twenty well-known polymers (polyethylene, polystyrene, PMMA, PTFE, …) carry rounded, widely quoted approximate T_g values; these are general scientific facts, not taken from any one source. The remaining 28 vinyl polymers receive **simulated** T_g values from a linear model fitted to those twenty, plus noise (σ = 15 °C). Always filter on the `source` column before drawing scientific conclusions.

## Data generated inside chapters

Many chapters build their data on the spot so that you can see and change every assumption. The main ones:

| Chapter | What is simulated |
|---|---|
| 12–13, 27, 39 | IR, UV–vis and NMR spectra from sums of Gaussian or Lorentzian bands |
| 15, 25 | kinetic runs from integrated rate laws with noise |
| 29 | Lennard-Jones molecular dynamics and Monte Carlo |
| 28, 40 | quantum-chemical energies computed with PySCF on small molecules (Chapter 40 builds its own small QM9-style set) |
| 42 | battery capacity-fade curves |
| 48 | an 8-sensor gas array with Langmuir response, ageing and drift |
| 49 | synthetic micrographs and a hyperspectral image cube |
| 50 | a batch reactor, on-line spectra and a deactivating catalyst |

QM9-style or other public datasets mentioned in the text (for example {cite:p}`ramakrishnan2014,wu2018`) are *cited* for context and are never redistributed with the book; where a chapter shows how to download one, the download comes from the original publisher under its own licence.
