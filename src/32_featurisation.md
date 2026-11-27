---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Featurising Molecules and Materials

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 20, 21 and 31.
```

A model can only learn what its features let it see. Machine-learning algorithms work on vectors of numbers, but molecules are graphs, materials are periodic crystals and samples are spectra. **Featurisation** (also called *representation* or *encoding*) is the step that turns chemistry into a fixed-length vector of numbers for each example. It is usually the most important decision in a chemical ML project: a sophisticated model given features that ignore the relevant chemistry will fail, while a simple model given the right features often succeeds. A good featurisation captures the structural information that drives the property, gives similar vectors to similar molecules, and works for every example in the dataset. This chapter compares descriptors, fingerprints, graphs, strings, composition features and spectra, and tests several of them on the same tasks.

The smallest featurisation of all is a hand-made list of numbers for each molecule, a few descriptors stacked into a matrix with one row per molecule:

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Descriptors
small = ["CCO", "c1ccccc1", "CC(=O)O"]
import numpy as np
X_small = np.array([[Descriptors.MolWt(Chem.MolFromSmiles(s)), Descriptors.MolLogP(Chem.MolFromSmiles(s))] for s in small])
print(X_small.round(2))          # rows: molecules; columns: MW, logP
```

Every featurisation in this chapter produces a matrix of this shape, just wider. We start by loading the dataset of Chapter 31.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, MACCSkeys, Draw

DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name)

df = load_data("molecules_sim.csv")
mols = [Chem.MolFromSmiles(s) for s in df.smiles]
```

## 1. Physicochemical descriptors

**Descriptors** (Chapter 20) are numbers with a physical or chemical meaning: molecular weight, logP, polar surface area, counts of rings and heteroatoms, topological indices. Computing all of RDKit's 200-odd descriptors gives a rich, interpretable feature vector. Two clean-up steps are routine. Some descriptors return infinity or NaN for unusual molecules, and `np.nan_to_num` replaces those with zero. Others have the same value for every molecule in the dataset, so they carry no information; dropping columns whose standard deviation is essentially zero removes them.

```{code-cell} ipython3
desc_names = [d[0] for d in Descriptors._descList]
X_desc = np.array([[f(m) for _, f in Descriptors._descList] for m in mols])
X_desc = np.nan_to_num(X_desc, posinf=0, neginf=0)
keep = X_desc.std(axis=0) > 1e-6
X_desc, desc_names = X_desc[:, keep], [n for n, k in zip(desc_names, keep) if k]
print(X_desc.shape, "descriptor matrix")
```

## 2. Fingerprints

**Fingerprints** (Chapter 21) record which substructures are present. For ML they must be converted from RDKit's bit-vector objects to NumPy arrays, which `DataStructs.ConvertToNumpyArray` does. Three variants are compared here. **Morgan bits** record presence or absence of each circular environment. **Morgan counts** record how many times each environment occurs, which distinguishes, for example, one hydroxyl from three. **MACCS keys** answer 166 predefined structural questions. Fingerprints are *sparse*: only a small fraction of bits are set for any one molecule, as the bit density shows.

```{code-cell} ipython3
def fp_matrix(gen_or_func, mols, n_bits=2048):
    out = np.zeros((len(mols), n_bits if n_bits else 167), dtype=np.uint8)
    for i, m in enumerate(mols):
        fp = gen_or_func(m)
        DataStructs.ConvertToNumpyArray(fp, out[i])
    return out

morgan = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X_morgan = fp_matrix(morgan.GetFingerprint, mols)
X_count = np.array([morgan.GetCountFingerprintAsNumPy(m) for m in mols])
X_maccs = fp_matrix(MACCSkeys.GenMACCSKeys, mols, n_bits=None)
print(X_morgan.shape, X_maccs.shape, f"Morgan bit density {X_morgan.mean():.3%}")
```

## 3. Compare them on two tasks

The fair way to compare featurisations is to keep everything else fixed (same model, same cross-validation folds, same metric) and change only the features. Here a random forest predicts the simulated activity (pIC₅₀, scored by R²) and toxicity (scored by ROC-AUC), with 5-fold cross-validation (Chapter 31).

```{code-cell} ipython3
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

feature_sets = {"5 simple descriptors": df[["MW", "logP", "TPSA", "HBD", "RotB"]].values,
                "all RDKit descriptors": X_desc, "MACCS keys": X_maccs, "Morgan bits": X_morgan, "Morgan counts": X_count}
results = []
for name, X in feature_sets.items():
    r2 = cross_val_score(RandomForestRegressor(200, n_jobs=-1, random_state=0), X, df.pIC50_sim,
                         cv=KFold(5, shuffle=True, random_state=0), scoring="r2").mean()
    auc = cross_val_score(RandomForestClassifier(200, n_jobs=-1, class_weight="balanced", random_state=0), X, df.toxic_sim,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), scoring="roc_auc").mean()
    results.append({"features": name, "pIC50 R²": r2, "toxicity ROC-AUC": auc})
res = pd.DataFrame(results).set_index("features")
res.plot.barh(subplots=True, layout=(1, 2), figsize=(11, 3.5), legend=False, color=["#6c4ad6", "#0f9d8a"], sharex=False)
plt.tight_layout(); plt.show()
res.round(3)
```

Every rich representation beats the five simple descriptors on both tasks. For activity the full set of RDKit descriptors does best here, and for toxicity the substructure-aware MACCS keys and Morgan fingerprints edge ahead, because the simulated toxicity is driven by **substructures** (toxic alerts) that whole-molecule descriptors such as MW or TPSA blur away. The differences between the rich representations are small, which is typical: the gap between a poor featurisation and a reasonable one is usually much larger than the gap between reasonable ones.

## 4. Graphs

Descriptors and fingerprints compress a molecule into one vector of fixed length, and some information is lost in the process. A **graph featurisation** keeps the molecule's full structure: an atom-feature matrix with one row of numbers per atom, and an adjacency matrix (Chapter 19) saying which atoms are bonded. Each atom's features are typically a **one-hot encoding** of its element (a list of 0s with a single 1 marking which element it is) plus properties such as aromaticity, hydrogen count, degree and charge. Because molecules have different numbers of atoms, these matrices have different sizes, so they need models designed for graphs, the graph neural networks of Chapter 34.

One-hot encoding is easiest to see on its own:

```{code-cell} ipython3
elements = ["C", "N", "O"]
for atom in ["N", "C", "O"]:
    print(atom, [int(atom == e) for e in elements])
```

```{code-cell} ipython3
ATOMS = ["C", "N", "O", "S", "F", "Cl", "Br", "other"]
def atom_features(a):
    sym = a.GetSymbol() if a.GetSymbol() in ATOMS else "other"
    one_hot = [sym == s for s in ATOMS]
    return np.array(one_hot + [a.GetIsAromatic(), a.GetTotalNumHs() / 3, a.GetDegree() / 4, a.GetFormalCharge(),
                               a.IsInRing()], dtype=float)

def mol_to_graph(m):
    Xn = np.array([atom_features(a) for a in m.GetAtoms()])
    A = Chem.GetAdjacencyMatrix(m).astype(float)
    return Xn, A

Xn, A = mol_to_graph(Chem.MolFromSmiles("Nc1ccncc1"))
fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), gridspec_kw={"width_ratios": [2, 1]})
sns.heatmap(Xn, ax=axes[0], cmap="Purples", cbar=False, xticklabels=ATOMS + ["arom", "H", "deg", "chg", "ring"])
axes[0].set(title="node features of 4-aminopyridine", ylabel="atom")
sns.heatmap(A, ax=axes[1], cmap="Greens", cbar=False, square=True); axes[1].set_title("adjacency")
plt.tight_layout(); plt.show()
```

## 5. Strings and tokens

Language models read SMILES as sequences of tokens. A **token** is the smallest meaningful unit of the string: a single atom such as `C`, a two-letter element such as `Cl` (which must not be split into `C` and `l`), a bracketed atom such as `[NH4+]`, a bond symbol or a ring-closure digit. **Tokenisation** splits a string into such units, and the set of all distinct tokens in a dataset is the **vocabulary**. Each token is then mapped to an integer, so a molecule becomes a sequence of integers that a sequence model such as a transformer can read (Chapters 45 and 46). A regular expression (Chapter 3) does the splitting.

```{code-cell} ipython3
import re
SMILES_TOKENS = re.compile(r"(\[[^\]]+]|Br|Cl|Si|@@|%\d{2}|[BCNOPSFIbcnops]|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\?|>|\*|\$|\d)")
def tokenize(smi):
    return SMILES_TOKENS.findall(smi)
print(tokenize("CC(=O)Oc1ccccc1C(=O)O"))
vocab = sorted({t for s in df.smiles for t in tokenize(s)})
print(len(vocab), "tokens in the vocabulary:", vocab)
```

**SELFIES** {cite:p}`krenn2020` is an alternative string grammar in which every string decodes to a valid molecule, which is handy for generative models (Chapter 45).

## 6. Composition features for materials

Materials often have no fixed molecular graph: an alloy or a mixed oxide is described by its formula, and the crystal structure may not be known. Composition-based features, such as averages and spreads of elemental properties, work surprisingly well {cite:p}`ward2016`.

The idea is to describe a compound by statistics of its elements' properties: the fraction-weighted **mean** electronegativity, the **range** of atomic radii, the average **deviation** from the mean, and so on. For NaCl, the mean electronegativity is simply the average of Na and Cl, weighted by their atomic fractions of 0.5 each:

```{code-cell} ipython3
from pymatgen.core import Composition, Element
print(0.5 * Element("Na").X + 0.5 * Element("Cl").X)
```

Applying six elemental properties and three statistics gives eighteen features per formula:

```{code-cell} ipython3

PROPS = {"Z": lambda e: e.Z, "X": lambda e: e.X, "radius": lambda e: float(e.atomic_radius or 0),
         "row": lambda e: e.row, "group": lambda e: e.group, "mass": lambda e: float(e.atomic_mass)}

def composition_features(formula):
    c = Composition(formula)
    fr = np.array([c.get_atomic_fraction(e) for e in c.elements])
    feats = {}
    for name, f in PROPS.items():
        v = np.array([f(e) for e in c.elements], dtype=float)
        mean = (fr * v).sum()
        feats[f"mean_{name}"] = mean
        feats[f"range_{name}"] = v.max() - v.min()
        feats[f"dev_{name}"] = (fr * abs(v - mean)).sum()
    return feats

pd.DataFrame({f: composition_features(f) for f in ["NaCl", "GaAs", "SrTiO3", "LiFePO4", "CsPbI3"]}).T.round(2).iloc[:, :9]
```

## 7. Spectra as features

A spectrum is already a vector. Binning it to a fixed grid makes it a feature vector for classification (Chapter 39). If every spectrum is recorded (or interpolated, Chapter 16) at the same wavenumbers, the absorbance at each wavenumber becomes one feature, and a model can learn which regions distinguish one class of compound from another. Practical issues are baseline drift and small shifts in peak positions between instruments, which is why spectra are usually pre-processed first (Chapter 27). Below, simulated IR spectra of an alcohol, a ketone and an acid differ in exactly the bands an organic chemist would look for.

```{code-cell} ipython3
wn = np.linspace(4000, 500, 700)
def fake_ir(groups, rng):
    bands = {"OH": (3350, 120, 0.7), "CH": (2950, 30, 0.5), "C=O": (1715, 18, 1.0), "C-O": (1150, 35, 0.6), "NH": (3300, 50, 0.5)}
    y = sum(h * np.exp(-0.5 * ((wn - c - rng.normal(0, 10)) / w) ** 2) for g in groups for c, w, h in [bands[g]])
    return y + rng.normal(0, 0.02, wn.size)
rng = np.random.default_rng(0)
spec = {"alcohol": fake_ir(["OH", "CH", "C-O"], rng), "ketone": fake_ir(["CH", "C=O"], rng), "acid": fake_ir(["OH", "CH", "C=O", "C-O"], rng)}
fig, ax = plt.subplots(figsize=(9, 3))
for (k, v), c in zip(spec.items(), ["#0f9d8a", "#e5487a", "#6c4ad6"]):
    ax.plot(wn, v, color=c, label=k)
ax.invert_xaxis(); ax.legend(); ax.set(xlabel="cm$^{-1}$", title="Each spectrum is a 700-dimensional feature vector")
plt.show()
```

## Choosing features: a summary

There is no single best representation; the right choice depends on the property, the amount of data and whether you need to interpret the model. As a rule of thumb, start with descriptors or fingerprints and a random forest, which form a strong baseline on small datasets, and move to learned representations such as graphs only when you have thousands of examples or more.

| Representation | Good for | Watch out |
|---|---|---|
| Descriptors | small data, interpretability | may miss local structure |
| Fingerprints | similarity, activity, toxicity | bit collisions, not invertible |
| Graphs | deep learning, large data | needs more data and compute |
| SMILES / SELFIES | language and generative models | many strings per molecule |
| Composition | inorganic materials | ignores structure and polymorphs |
| 3D / crystal graphs | quantum and materials properties | need geometries |
| Spectra | analytical chemistry | baseline, alignment, resolution |

```{admonition} Summary
:class: tip
- Featurisation turns each molecule, material or sample into a fixed-length vector; it often matters more than the choice of model.
- Descriptors are interpretable; clean out NaNs and constant columns.
- Fingerprints capture substructures, which matter for properties such as toxicity; counts add multiplicity.
- Compare representations with the same model, folds and metric.
- Graphs (atom features + adjacency) feed graph neural networks; tokenised SMILES feed language models.
- Composition statistics describe inorganic materials; binned spectra are features for analytical problems.
```

## Exercises

**★ 1.** How many Morgan bits are set, on average, at radius 1, 2 and 3?

**★★ 2.** Find the 10 descriptors most correlated with `logS_sim`. Are any of them redundant?

**★★★ 3.** Add composition features to the binary-materials dataset and compare a random forest predicting `band_gap_sim_eV` with and without them.
