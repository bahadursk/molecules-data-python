---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Representing Molecules: SMILES, InChI and Molfiles

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Part I and general chemistry. RDKit is introduced gently here and fully in Chapter 20.
```

A computer cannot look at a skeletal formula. It needs a precise, machine-readable description of a molecule. This chapter compares the main ways of writing molecules down: graphs, SMILES, InChI and connection tables.

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
```

## Molecules as graphs

To a chemist, a molecule is atoms joined by bonds. To a mathematician, that is a **graph**: nodes (atoms) joined by edges (bonds). Hydrogens are usually left implicit.

```{code-cell} ipython3
m = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")     # aspirin
d = rdMolDraw2D.MolDraw2DSVG(420, 300)
d.drawOptions().addAtomIndices = True
d.DrawMolecule(m); d.FinishDrawing()
SVG(d.GetDrawingText())
```

```{code-cell} ipython3
import numpy as np
A = Chem.GetAdjacencyMatrix(m)
print("adjacency matrix shape:", A.shape)
print(A[:6, :6])
print("degree of each atom:", A.sum(axis=1))
```

The **adjacency matrix** is exactly what graph neural networks (Chapter 34) learn from.

## SMILES grammar

| Rule | Example | Molecule |
|---|---|---|
| Atoms are element symbols; H is implicit | `C`, `CCO` | methane, ethanol |
| `=` double, `#` triple bond | `C=C`, `C#N` | ethene, HCN |
| Branches in brackets | `CC(C)C` | isobutane |
| Ring closures with digits | `C1CCCCC1` | cyclohexane |
| Lowercase = aromatic | `c1ccccc1` | benzene |
| Charges and isotopes in square brackets | `[NH4+]`, `[13CH4]` | ammonium, ¹³C-methane |
| Stereo: `/ \` for E/Z, `@ @@` for R/S | `F/C=C/F`, `N[C@@H](C)C(=O)O` | *trans*-difluoroethene, L-alanine |
| `.` separates components | `[Na+].[Cl-]` | sodium chloride |

```{code-cell} ipython3
examples = {"isobutane": "CC(C)C", "cyclohexene": "C1CC=CCC1", "pyridine": "c1ccncc1",
            "ammonium": "[NH4+]", "L-alanine": "N[C@@H](C)C(=O)O", "trans-2-butene": "C/C=C/C",
            "cis-2-butene": "C/C=C\\C", "naphthalene": "c1ccc2ccccc2c1", "sodium acetate": "CC(=O)[O-].[Na+]",
            "glucose": "OC[C@H]1OC(O)[C@H](O)[C@@H](O)[C@@H]1O", "nicotine": "CN1CCC[C@H]1c1cccnc1",
            "cubane": "C12C3C4C1C5C2C3C45"}
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in examples.values()], legends=list(examples),
                     molsPerRow=4, subImgSize=(200, 170))
```

## One molecule, many SMILES

```{code-cell} ipython3
variants = ["OCC", "CCO", "C(O)C", "[CH3][CH2][OH]"]
print([Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in variants])
```

All four strings describe ethanol. **Canonical SMILES** gives one unique string per molecule, which makes it a database key and lets you find duplicates.

```{code-cell} ipython3
for s in ["C1=CC=CC=C1", "c1ccccc1"]:
    print(f"{s:14s} -> {Chem.MolToSmiles(Chem.MolFromSmiles(s))}")
```

RDKit **perceives aromaticity**, so the Kekulé form and the aromatic form give the same canonical SMILES.

## Tautomers and resonance

```{code-cell} ipython3
from rdkit.Chem.MolStandardize import rdMolStandardize
enumerator = rdMolStandardize.TautomerEnumerator()
tauts = list(enumerator.Enumerate(Chem.MolFromSmiles("Oc1ccccn1")))     # 2-hydroxypyridine
Draw.MolsToGridImage(tauts, legends=[Chem.MolToSmiles(t) for t in tauts], subImgSize=(220, 170))
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
2-Hydroxypyridine and 2-pyridone are tautomers: different molecules in rapid equilibrium. Their SMILES differ, so a database search for one can miss the other. Standardising tautomers is an essential cleaning step before machine learning.
```

## InChI and InChIKey

The IUPAC **International Chemical Identifier** is a layered, canonical, non-proprietary identifier. The **InChIKey** is a fixed-length 27-character hash, ideal for web searches.

```{code-cell} ipython3
for name in ["L-alanine", "glucose", "nicotine"]:
    mol = Chem.MolFromSmiles(examples[name])
    print(name)
    print("  InChI   :", Chem.MolToInchi(mol))
    print("  InChIKey:", Chem.MolToInchiKey(mol))
```

The InChI layers are separated by `/`: formula, connections (`c`), hydrogens (`h`), stereochemistry (`t`, `m`, `s`) and more.

## Molfiles and connection tables

A **molfile** lists atoms with coordinates and a bond table. SDF files (Chapter 8) are molfiles plus data fields.

```{code-cell} ipython3
ethanol = Chem.MolFromSmiles("CCO")
AllChem.Compute2DCoords(ethanol)
print(Chem.MolToMolBlock(ethanol))
```

## 3D representations

```{code-cell} ipython3
mh = Chem.AddHs(ethanol)
AllChem.EmbedMolecule(mh, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mh)
print(Chem.MolToXYZBlock(mh))
```

## Other notations you will meet

| Notation | Typical use |
|---|---|
| **SMARTS** | substructure *patterns* (Chapter 21) |
| **Reaction SMILES / SMARTS** | reactions (Chapter 22) |
| **SELFIES** | always-valid strings for generative ML (Chapter 45) |
| **BigSMILES** | polymers (Chapter 43) |
| **CIF** | crystal structures (Chapter 30) |
| **PDB / mmCIF** | proteins and nucleic acids (Chapter 23) |
| **IUPAC names** | humans |

## Comparing representations

```{code-cell} ipython3
import pandas as pd
rows = []
for name, smi in examples.items():
    mol = Chem.MolFromSmiles(smi)
    rows.append({"name": name, "canonical SMILES": Chem.MolToSmiles(mol), "InChIKey": Chem.MolToInchiKey(mol),
                 "formula": Chem.rdMolDescriptors.CalcMolFormula(mol)})
pd.DataFrame(rows)
```

## Exercises

**★ 1.** Write SMILES for propanoic acid, cyclopentanone and toluene, and draw them.

**★★ 2.** Show that `N[C@@H](C)C(=O)O` and `N[C@H](C)C(=O)O` give different InChIKeys. Which layer differs?

**★★★ 3.** Enumerate the tautomers of guanine (`NC1=Nc2[nH]cnc2C(=O)N1`) and count them.
