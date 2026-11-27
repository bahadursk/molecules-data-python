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

A chemist glancing at a skeletal formula sees at once which atoms are bonded, where the double bonds are and which carbon is a stereocentre. A computer cannot look at a drawing. It needs a precise, unambiguous, machine-readable description of the molecule, and the choice of description matters: some are compact enough to type, some give exactly one string per molecule so that databases can find duplicates, some store 3D coordinates, and some are designed for machine learning. Every cheminformatics task, from a database search to a property prediction, starts by choosing how to represent the molecules. This chapter compares the main ways of writing molecules down: graphs, SMILES, InChI and connection tables, and shows how RDKit converts between them.

The `rdkit.Chem` module holds RDKit's core molecule functions, `Draw` makes pictures, and `AllChem` adds coordinate generation:

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
```

## Molecules as graphs

To a chemist, a molecule is atoms joined by bonds. To a mathematician, that is a **graph**: a set of nodes (atoms) joined by edges (bonds). Each node carries labels such as the element and charge, and each edge carries a bond order. A graph says nothing about where the atoms are in space, only what is connected to what, which is exactly the information in a skeletal formula. Hydrogens are usually left implicit, because they can be worked out from the valences of the heavy atoms.

`Chem.MolFromSmiles` reads a SMILES string (Section 19.2) and returns an RDKit `Mol` object, which stores the graph. The simplest questions to ask it are how many atoms and bonds it has:

```{code-cell} ipython3
m = Chem.MolFromSmiles("CCO")       # ethanol
print(m.GetNumAtoms(), "heavy atoms,", m.GetNumBonds(), "bonds")
for atom in m.GetAtoms():
    print(atom.GetIdx(), atom.GetSymbol(), "with", atom.GetTotalNumHs(), "H")
```

Drawing a molecule with atom indices shows how RDKit numbers the nodes of the graph:

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

A graph can be stored as a table of numbers called the **adjacency matrix**: a square grid with one row and one column per atom, holding 1 where two atoms are bonded and 0 elsewhere. Summing a row gives the atom's **degree**, its number of heavy-atom neighbours. The adjacency matrix is exactly what graph neural networks (Chapter 34) learn from.

## SMILES grammar

**SMILES** (Simplified Molecular Input Line Entry System) writes a molecular graph as a single line of text by walking through the atoms. Chapter 3 introduced it; here are the rules in full. Its great advantages are that it is short, human-readable with practice, and easy to store in a spreadsheet column or a database. A handful of rules covers almost every organic molecule:

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

The best way to learn the grammar is to write a SMILES string, draw it, and check that the picture is what you meant. If a string is invalid, for example because a ring is never closed, `MolFromSmiles` prints a warning and returns `None`:

```{code-cell} ipython3
print(Chem.MolFromSmiles("C1CCCC") is None)     # ring 1 is opened but never closed
```

Here is a gallery of examples covering each rule:

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

A SMILES string depends on where you start walking through the graph and which branch you take first, so one molecule can be written in many valid ways. That is a problem for a database: if one chemist stores ethanol as `CCO` and another as `OCC`, a simple text search will treat them as different compounds. The cure is to convert every structure to a standard form. `Chem.MolToSmiles` writes RDKit's **canonical SMILES**, produced by a fixed algorithm that numbers the atoms in a unique way.

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

Canonical SMILES solves the "same graph, different string" problem, but chemistry adds a subtler one. **Resonance** structures are different ways of drawing the electrons of the *same* molecule; RDKit handles the most important case, aromaticity, as shown above. **Tautomers** are different molecules, usually related by moving a hydrogen and a double bond, that interconvert rapidly in solution: the keto and enol forms of a ketone, or 2-hydroxypyridine and 2-pyridone. Because tautomers have different graphs, they get different canonical SMILES. RDKit's `TautomerEnumerator` lists the tautomers of a molecule and can choose a standard one.

```{code-cell} ipython3
keto, enol = Chem.MolFromSmiles("CC(=O)C"), Chem.MolFromSmiles("CC(O)=C")   # acetone and its enol
print(Chem.MolToSmiles(keto), Chem.MolToSmiles(enol))
```

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

Canonical SMILES depend on the software that produced them: RDKit and another toolkit may write different canonical strings for the same molecule. The IUPAC **International Chemical Identifier** (InChI) solves this by being a single open standard, with one official algorithm, so that every program produces the same identifier for the same compound. It is built in *layers*, each adding information: formula, connectivity, hydrogens, charge, stereochemistry. It also standardises some tautomers automatically. InChI strings are long, so the **InChIKey** compresses one into a fixed-length 27-character code by hashing. The key cannot be turned back into a structure, but it is ideal for web searches and for linking records between databases.

```{code-cell} ipython3
print(Chem.MolToInchi(Chem.MolFromSmiles("CCO")))
print(Chem.MolToInchiKey(Chem.MolFromSmiles("OCC")))      # same key whichever SMILES you start from
```

```{code-cell} ipython3
for name in ["L-alanine", "glucose", "nicotine"]:
    mol = Chem.MolFromSmiles(examples[name])
    print(name)
    print("  InChI   :", Chem.MolToInchi(mol))
    print("  InChIKey:", Chem.MolToInchiKey(mol))
```

The InChI layers are separated by `/`: formula, connections (`c`), hydrogens (`h`), stereochemistry (`t`, `m`, `s`) and more.

## Molfiles and connection tables

A **molfile** (MDL Molfile) is the oldest widely used structure format and the one most drawing programs export. It is a **connection table**: a header, a "counts line" giving the number of atoms and bonds, an *atom block* with one line per atom (x, y, z coordinates and element), and a *bond block* with one line per bond (the two atom numbers and the bond order). Because it stores coordinates, it can record how a molecule was drawn or its 3D shape, which a SMILES cannot. SDF files (Chapter 8) are molfiles plus data fields. `Compute2DCoords` generates a tidy 2D layout before writing.

```{code-cell} ipython3
ethanol = Chem.MolFromSmiles("CCO")
AllChem.Compute2DCoords(ethanol)
print(Chem.MolToMolBlock(ethanol))
```

## 3D representations

Properties such as shape, dipole moment and how a drug fits a protein depend on the molecule's three-dimensional geometry. A 3D structure is generated from the graph in three steps: add the hydrogens explicitly (`AddHs`), *embed* the molecule by generating plausible 3D coordinates from distance rules (`EmbedMolecule`, with a random seed for reproducibility), and then relax the geometry with a force field (`MMFFOptimizeMolecule`). The result can be written in the XYZ format of Chapter 8; Chapter 23 covers 3D structures in depth.

```{code-cell} ipython3
mh = Chem.AddHs(ethanol)
AllChem.EmbedMolecule(mh, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mh)
print(Chem.MolToXYZBlock(mh))
```

## Other notations you will meet

Specialised fields have their own formats, each designed for something SMILES does poorly. You will meet these later in the book:

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

Putting the representations side by side in a pandas table (Chapter 15) shows their character: canonical SMILES are short and readable, InChIKeys are fixed-length and opaque, and molecular formulas lose all structural information, so isomers share them.

```{code-cell} ipython3
import pandas as pd
rows = []
for name, smi in examples.items():
    mol = Chem.MolFromSmiles(smi)
    rows.append({"name": name, "canonical SMILES": Chem.MolToSmiles(mol), "InChIKey": Chem.MolToInchiKey(mol),
                 "formula": Chem.rdMolDescriptors.CalcMolFormula(mol)})
pd.DataFrame(rows)
```

```{admonition} Summary
:class: tip
- A molecule is a graph of atoms and bonds; the adjacency matrix stores it as numbers.
- SMILES writes the graph as text; many SMILES describe one molecule, so use **canonical SMILES** as a key.
- Tautomers are different molecules with different SMILES; standardise them before searching or modelling.
- InChI is the open, layered standard identifier; the InChIKey is its 27-character hash for searching.
- Molfiles and SDF store connection tables with coordinates; 3D structures come from embedding plus force-field optimisation.
```

## Exercises

**★ 1.** Write SMILES for propanoic acid, cyclopentanone and toluene, and draw them.

**★★ 2.** Show that `N[C@@H](C)C(=O)O` and `N[C@H](C)C(=O)O` give different InChIKeys. Which layer differs?

**★★★ 3.** Enumerate the tautomers of guanine (`NC1=Nc2[nH]cnc2C(=O)N1`) and count them.
