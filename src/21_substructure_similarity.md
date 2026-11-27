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

"Find every compound containing a carboxylic acid." "Which molecules in my library look like this hit?" These two questions, **substructure search** and **similarity search**, are the bread and butter of cheminformatics.

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

```{code-cell} ipython3
benzene_ring = Chem.MolFromSmiles("c1ccccc1")
hits = [name for name, m in mols.items() if m.HasSubstructMatch(benzene_ring)]
print(len(hits), "of", len(mols), "contain a benzene ring")
```

## SMARTS: a pattern language

SMARTS extends SMILES with wildcards and logic.

| SMARTS | Meaning |
|---|---|
| `[#6]` | any carbon (aromatic or not) |
| `[OX2H]` | oxygen with two connections, one of them H (hydroxyl) |
| `[CX3](=O)[OX2H1]` | carboxylic acid |
| `[NX3;H2,H1;!$(NC=O)]` | primary or secondary amine, not an amide |
| `c[Cl,Br,I]` | aryl halide |
| `[R]` / `[!R]` | ring / non-ring atom |
| `*` | any atom |

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

```{code-cell} ipython3
acid = patterns["carboxylic acid"]
acids = [m for m in mols.values() if m.HasSubstructMatch(acid)]
Draw.MolsToGridImage(acids, highlightAtomLists=[m.GetSubstructMatch(acid) for m in acids],
                     legends=[k for k, m in mols.items() if m.HasSubstructMatch(acid)], molsPerRow=4, subImgSize=(200, 160))
```

### Maximum common substructure

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

A **fingerprint** turns a molecule into a fixed-length bit vector. The most widely used is the **Morgan** (ECFP-like) fingerprint {cite:p}`rogers2010`, which records the circular neighbourhood of every atom up to a given radius.

```{code-cell} ipython3
mfp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = {k: mfp.GetFingerprint(m) for k, m in mols.items()}
fp = fps["aspirin"]
print(fp.GetNumBits(), "bits,", fp.GetNumOnBits(), "set:", list(fp.GetOnBits())[:12], "...")
```

### What does a bit mean?

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

$$T(A, B) = \frac{|A \cap B|}{|A \cup B|}$$

```{code-cell} ipython3
names = list(fps)
S = np.array([DataStructs.BulkTanimotoSimilarity(fps[a], [fps[b] for b in names]) for a in names])
plt.figure(figsize=(9, 7.5))
sns.heatmap(pd.DataFrame(S, index=names, columns=names), cmap="magma", vmin=0, vmax=1, square=True)
plt.title("Tanimoto similarity (Morgan r=2)"); plt.show()
```

Clear families appear: the profens, the xanthines, the opioids and the local anaesthetics.

### Similarity search

```{code-cell} ipython3
query = Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O")      # (S)-ibuprofen as query
qfp = mfp.GetFingerprint(query)
ranked = sorted(((DataStructs.TanimotoSimilarity(qfp, f), n) for n, f in fps.items()), reverse=True)
for sim, n in ranked[:5]:
    print(f"{sim:.2f}  {n}")
```

### Other fingerprints

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

Which atoms make codeine similar to morphine? Remove each atom in turn and see how much similarity is lost.

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

## Exercises

**★ 1.** Write a SMARTS pattern for a nitro group and test it on nitrobenzene and TNT.

**★★ 2.** Find the three molecules in the library most similar to lidocaine using MACCS keys.

**★★★ 3.** Implement Tanimoto similarity yourself with NumPy bit arrays and verify it against `DataStructs`.
