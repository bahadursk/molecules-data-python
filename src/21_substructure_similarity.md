---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# RDKit II: Substructures, SMARTS and Fingerprints

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 19–20.
```

"Find every compound containing a carboxylic acid." "Which molecules in my library look like this hit?" These two questions, **substructure search** and **similarity search**, are the bread and butter of cheminformatics. A substructure search asks a yes/no question, "does this molecule contain that fragment?", and is used to filter libraries, flag reactive or toxic groups, and count functional groups. A similarity search asks "how alike are these two molecules?", and rests on the idea that structurally similar molecules tend to have similar properties, which is why it guides medicinal chemists from a first hit to related compounds worth testing. Both must work on thousands or millions of molecules in seconds. This chapter shows how RDKit answers each: SMARTS patterns for substructures, and fingerprints with the Tanimoto coefficient for similarity.

```{code-cell} ipython3
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem, MACCSkeys, rdFingerprintGenerator
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
```

## A small library

To have something to search, we build a small **library**, a collection of molecules, here sixteen familiar drugs from several families: anti-inflammatory profens, xanthine stimulants, opioids and local anaesthetics. A dictionary maps each name to its molecule object.

```{code-cell} ipython3
library = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O", "salicylic acid": "OC(=O)c1ccccc1O", "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "naproxen": "COc1ccc2cc([C@H](C)C(=O)O)ccc2c1", "ketoprofen": "CC(C(=O)O)c1cccc(c1)C(=O)c1ccccc1",
    "diclofenac": "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl", "paracetamol": "CC(=O)Nc1ccc(O)cc1",
    "phenacetin": "CCOc1ccc(NC(C)=O)cc1", "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "theobromine": "CN1C=NC2=C1C(=O)NC(=O)N2C", "theophylline": "CN1C2=C(C(=O)N(C1=O)C)NC=N2",
    "morphine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "codeine": "CN1CC[C@]23c4c5ccc(OC)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "benzocaine": "CCOC(=O)c1ccc(N)cc1", "procaine": "CCN(CC)CCOC(=O)c1ccc(N)cc1",
    "lidocaine": "CCN(CC)CC(=O)Nc1c(C)cccc1C",
}
mols = {k: Chem.MolFromSmiles(v) for k, v in library.items()}
Draw.MolsToGridImage(list(mols.values()), legends=list(mols), molsPerRow=4, subImgSize=(200, 160))
```

## Substructure search

A **substructure** is a piece of a molecule: a ring, a functional group or a larger fragment. Mathematically, a substructure search checks whether the fragment's graph can be laid over part of the molecule's graph, atom for atom and bond for bond (a *subgraph isomorphism*). `mol.HasSubstructMatch(fragment)` answers yes or no, and `mol.GetSubstructMatches(fragment)` returns the atom indices of every place it fits. The simplest query is a molecule written as SMILES:

```{code-cell} ipython3
toluene = Chem.MolFromSmiles("Cc1ccccc1")
benzene = Chem.MolFromSmiles("c1ccccc1")
print(toluene.HasSubstructMatch(benzene))      # toluene contains a benzene ring
print(Chem.MolFromSmiles("CCO").GetSubstructMatches(Chem.MolFromSmiles("C")))   # every carbon
```

Searching the whole library for a benzene ring is a one-line comprehension:

```{code-cell} ipython3
benzene_ring = Chem.MolFromSmiles("c1ccccc1")
hits = [name for name, m in mols.items() if m.HasSubstructMatch(benzene_ring)]
print(len(hits), "of", len(mols), "contain a benzene ring")
```

## SMARTS: a pattern language

A query written as SMILES is very literal: `C=O` matches only a carbon doubly bonded to an oxygen with nothing said about the neighbours. Real questions are more flexible: "an oxygen with exactly one hydrogen", "any halogen on an aromatic ring", "a nitrogen that is not part of an amide". **SMARTS** (SMILES Arbitrary Target Specification) extends SMILES with wildcards and logic to express such patterns. Inside square brackets you can list atom properties: `#6` means atomic number 6 (any carbon), `X2` means two connections, `H1` one hydrogen, `R` in a ring. A comma means *or*, a semicolon or ampersand means *and*, `!` means *not*, and `$(...)` means "an atom that is the start of this sub-pattern". Every valid SMILES is also a valid SMARTS, so you can start simple and add conditions.

| SMARTS | Meaning |
|---|---|
| `[#6]` | any carbon (aromatic or not) |
| `[OX2H]` | oxygen with two connections, one of them H (hydroxyl) |
| `[CX3](=O)[OX2H1]` | carboxylic acid |
| `[NX3;H2,H1;!$(NC=O)]` | primary or secondary amine, not an amide |
| `c[Cl,Br,I]` | aryl halide |
| `[R]` / `[!R]` | ring / non-ring atom |
| `*` | any atom |

The difference between a SMILES query and a SMARTS query shows up immediately. `C(=O)O` written as SMILES-style SMARTS matches esters as well as acids, because it does not say whether the second oxygen carries a hydrogen; adding `[OX2H1]` restricts it to acids:

```{code-cell} ipython3
loose = Chem.MolFromSmarts("C(=O)O")
strict = Chem.MolFromSmarts("[CX3](=O)[OX2H1]")
for name in ["aspirin", "benzocaine"]:
    print(f"{name:10s} loose: {mols[name].HasSubstructMatch(loose)}  acid: {mols[name].HasSubstructMatch(strict)}")
```

Benzocaine is an ester, not an acid, and only the stricter pattern gets it right. A dictionary of patterns turns the whole library into a table of 0/1 values, one column per functional group, shown here as a heatmap:

```{code-cell} ipython3
functional_groups = {
    "carboxylic acid": "[CX3](=O)[OX2H1]", "ester": "[#6][CX3](=O)[OX2][#6]", "amide": "[NX3][CX3](=[OX1])",
    "phenol": "c[OX2H]", "aryl ether": "c[OX2][CX4]", "aryl halide": "c[Cl,Br,I,F]",
    "primary arylamine": "c[NX3H2]", "tertiary amine": "[NX3;H0;!$(NC=O);!$(N-a)]([#6])([#6])[#6]",
    "ketone": "[#6][CX3](=O)[#6]",
}
patterns = {k: Chem.MolFromSmarts(v) for k, v in functional_groups.items()}
table = pd.DataFrame({fg: [int(m.HasSubstructMatch(p)) for m in mols.values()] for fg, p in patterns.items()},
                     index=list(mols))
plt.figure(figsize=(9, 6))
sns.heatmap(table, cmap=sns.color_palette(["#f4f4f8", "#6c4ad6"]), linewidths=0.5, cbar=False, annot=True)
plt.title("Functional-group fingerprint of the library"); plt.show()
```

### Highlighting all matches

The atom indices from a match are exactly what the drawing functions need, so results can be checked by eye. `highlightAtomLists` takes one list of indices per molecule:

```{code-cell} ipython3
acid = patterns["carboxylic acid"]
acids = [m for m in mols.values() if m.HasSubstructMatch(acid)]
Draw.MolsToGridImage(acids, highlightAtomLists=[m.GetSubstructMatch(acid) for m in acids],
                     legends=[k for k, m in mols.items() if m.HasSubstructMatch(acid)], molsPerRow=4, subImgSize=(200, 160))
```

### Maximum common substructure

Sometimes you do not know the fragment in advance and want to discover what a set of molecules share. The **maximum common substructure** (MCS) is the largest fragment present in every molecule of a set. It reveals the common scaffold of a compound series. `rdFMCS.FindMCS` searches for it and returns it as a SMARTS string; for the three xanthine stimulants it is the shared purine-dione core.

```{code-cell} ipython3
from rdkit.Chem import rdFMCS
xanthines = [mols["caffeine"], mols["theobromine"], mols["theophylline"]]
mcs = rdFMCS.FindMCS(xanthines)
core = Chem.MolFromSmarts(mcs.smartsString)
print(mcs.numAtoms, "atoms in common:", mcs.smartsString)
Draw.MolsToGridImage(xanthines, highlightAtomLists=[m.GetSubstructMatch(core) for m in xanthines],
                     legends=["caffeine", "theobromine", "theophylline"], subImgSize=(220, 180))
```

## Molecular fingerprints

Substructure search is exact but slow on huge libraries, and it cannot say *how* similar two molecules are. A **fingerprint** solves both problems by turning a molecule into a fixed-length bit vector, a long list of 0s and 1s in which each bit records whether a particular structural feature is present. Two molecules can then be compared by comparing their bit lists, which a computer does extremely fast, and the same vectors serve as input to machine-learning models (Chapter 32). The most widely used is the **Morgan** (ECFP-like) fingerprint {cite:p}`rogers2010`, which records the circular neighbourhood of every atom up to a given radius. For each atom it describes the atom itself (radius 0), then the atom with its direct neighbours (radius 1), then out to two bonds (radius 2), turns each description into a number by *hashing*, and sets the corresponding bit in a vector of, typically, 2048 bits. Radius 2 gives the ECFP4 fingerprint widely used in drug discovery. A *generator* object holds these settings and is reused for every molecule.

A tiny molecule shows the idea: ethanol has only a handful of distinct atom environments, so only a few bits are set.

```{code-cell} ipython3
mfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
ethanol_fp = mfp.GetFingerprint(Chem.MolFromSmiles("CCO"))
print(ethanol_fp.GetNumOnBits(), "bits set out of", ethanol_fp.GetNumBits())
```

```{code-cell} ipython3
fps = {k: mfp.GetFingerprint(m) for k, m in mols.items()}
fp = fps["aspirin"]
print(fp.GetNumBits(), "bits,", fp.GetNumOnBits(), "set:", list(fp.GetOnBits())[:12], "...")
```

### What does a bit mean?

Each set bit corresponds to one or more atom environments, and it helps to see them. Asking the generator for "additional output" records which atom and radius produced each bit, and `Draw.DrawMorganBits` draws the fragment behind each bit, with the central atom highlighted. Because hashing squeezes millions of possible environments into 2048 bits, two unrelated environments occasionally share a bit (a *collision*), which is a small source of noise.

```{code-cell} ipython3
ao = rdFingerprintGenerator.AdditionalOutput()
ao.AllocateBitInfoMap()
mfp.GetFingerprint(mols["aspirin"], additionalOutput=ao)
bit_info = ao.GetBitInfoMap()
some_bits = list(bit_info)[:8]
Draw.DrawMorganBits([(mols["aspirin"], b, bit_info) for b in some_bits], molsPerRow=4,
                    legends=[f"bit {b}" for b in some_bits], subImgSize=(170, 150))
```

## Tanimoto similarity

Given two fingerprints, the **Tanimoto coefficient** (also called the Jaccard index) measures their overlap: the number of bits set in both, divided by the number of bits set in either.

$$T(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

It runs from 0 (no features in common) to 1 (identical fingerprints, which usually means identical molecules). Its meaning is easy to see on two short bit lists, computed by hand with sets of "on" positions:

```{code-cell} ipython3
A_on = {1, 4, 7, 9}             # positions of the 1-bits in fingerprint A
B_on = {1, 4, 8}
print(len(A_on & B_on) / len(A_on | B_on))     # 2 shared out of 5 in total = 0.4
```

`DataStructs.TanimotoSimilarity(fp1, fp2)` does the same for RDKit fingerprints, and `BulkTanimotoSimilarity` compares one fingerprint with a whole list at once. Computing every pair gives a **similarity matrix**:

```{code-cell} ipython3
names = list(fps)
S = np.array([DataStructs.BulkTanimotoSimilarity(fps[a], [fps[b] for b in names]) for a in names])
plt.figure(figsize=(9, 7.5))
sns.heatmap(pd.DataFrame(S, index=names, columns=names), cmap="magma", vmin=0, vmax=1, square=True)
plt.title("Tanimoto similarity (Morgan r=2)"); plt.show()
```

Clear families appear: the profens, the xanthines, the opioids and the local anaesthetics.

### Similarity search

A **similarity search** ranks a library by similarity to a query molecule, typically an active compound, so that the closest analogues can be tested next. It is the computational version of "find me more like this". Here (S)-ibuprofen is the query; the other profens rank highest because they share the aryl-propionic-acid motif.

```{code-cell} ipython3
query = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")      # (S)-ibuprofen as query
qfp = mfp.GetFingerprint(query)
ranked = sorted(((DataStructs.TanimotoSimilarity(qfp, f), n) for n, f in fps.items()), reverse=True)
for sim, n in ranked[:5]:
    print(f"{sim:.2f}  {n}")
```

### Other fingerprints

Morgan fingerprints are not the only kind, and each captures structure differently. **MACCS keys** are 166 predefined yes/no questions about the structure ("is there a sulfur?", "is there a six-membered ring?"), so every bit has a fixed, readable meaning. **Atom-pair** fingerprints record pairs of atoms together with the number of bonds between them, which captures overall shape. **RDKit path** fingerprints record every linear path of bonds up to a given length. Comparing the same molecule pairs with each type shows how much the answer depends on the choice:

```{code-cell} ipython3
fp_types = {
    "Morgan r2": lambda m: mfp.GetFingerprint(m),
    "MACCS": MACCSkeys.GenMACCSKeys,
    "AtomPair": lambda m: rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048).GetFingerprint(m),
    "RDKit path": lambda m: rdFingerprintGenerator.GetRDKitFPGenerator(fpSize=2048).GetFingerprint(m),
}
pairs = [("caffeine", "theobromine"), ("aspirin", "salicylic acid"), ("morphine", "codeine"), ("aspirin", "caffeine")]
res = pd.DataFrame({ft: [DataStructs.TanimotoSimilarity(f(mols[a]), f(mols[b])) for a, b in pairs]
                    for ft, f in fp_types.items()}, index=[f"{a} / {b}" for a, b in pairs]).round(2)
res.plot.bar(figsize=(8, 4), color=["#6c4ad6", "#0f9d8a", "#f2a93b", "#e5487a"], rot=15)
plt.ylabel("Tanimoto"); plt.title("Similarity depends on the fingerprint"); plt.show()
```

```{warning}
There is no single "true" similarity. A Tanimoto of 0.4 can be high for Morgan fingerprints and low for MACCS keys. Always state the fingerprint you used.
```

## Similarity maps

A Tanimoto value is a single number, but a chemist wants to know *which parts* of two molecules make them alike or different. A **similarity map** answers this by removing each atom's contribution to the fingerprint in turn and measuring how much the similarity to a reference drops. Atoms whose removal lowers the similarity are shared features; atoms whose removal raises it are differences. Which atoms make codeine similar to morphine? Remove each atom in turn and see how much similarity is lost.

```{code-cell} ipython3
def fp_func(m, atomId=-1):
    return SimilarityMaps.GetMorganFingerprint(m, atomId=atomId, radius=2, nBits=2048)
d = rdMolDraw2D.MolDraw2DCairo(450, 350)
_, maxw = SimilarityMaps.GetSimilarityMapForFingerprint(mols["morphine"], mols["codeine"], fp_func, draw2d=d)
d.FinishDrawing()
from IPython.display import Image
Image(d.GetDrawingText())
```

Green atoms increase similarity to the reference; pink atoms (the O-methyl group) decrease it.

```{admonition} Summary
:class: tip
- `HasSubstructMatch` and `GetSubstructMatches` find fragments; matches give atom indices for highlighting.
- SMARTS adds wildcards and logic (`#6`, `X2`, `H1`, `,` `;` `!`, `$()`), so patterns can be exact about functional groups.
- `rdFMCS.FindMCS` finds the largest fragment shared by a set of molecules.
- Fingerprints are bit vectors of structural features; Morgan (ECFP) fingerprints hash circular atom environments.
- Tanimoto similarity = shared bits / total bits; always state which fingerprint you used.
- Similarity maps show which atoms drive similarity.
```

## Exercises

**★ 1.** Write a SMARTS pattern for a nitro group and test it on nitrobenzene and TNT.

**★★ 2.** Find the three molecules in the library most similar to lidocaine using MACCS keys.

**★★★ 3.** Implement Tanimoto similarity yourself with NumPy bit arrays and verify it against `DataStructs`.
