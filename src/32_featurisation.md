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

A model can only learn what its features let it see. **Featurisation**, turning chemistry into numbers, is usually the most important decision in a chemical ML project. This chapter compares descriptors, fingerprints, graphs, strings, composition features and spectra.

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

```{code-cell} ipython3
desc_names = [d[0] for d in Descriptors._descList]
X_desc = np.array([[f(m) for _, f in Descriptors._descList] for m in mols])
X_desc = np.nan_to_num(X_desc, posinf=0, neginf=0)
keep = X_desc.std(axis=0) > 1e-6
X_desc, desc_names = X_desc[:, keep], [n for n, k in zip(desc_names, keep) if k]
print(X_desc.shape, "descriptor matrix")
```

## 2. Fingerprints

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

Structure-based fingerprints win for both tasks, because the simulated activity and toxicity depend on **substructures** (a hinge-binding motif, toxic alerts) that whole-molecule descriptors blur away.

## 4. Graphs

A graph featurisation stores an atom-feature matrix and an adjacency matrix: the input to graph neural networks (Chapter 34).

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

Language models read SMILES as sequences of tokens.

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

Materials often have no fixed molecular graph. Composition-based features, such as averages and spreads of elemental properties, work surprisingly well {cite:p}`ward2016`.

```{code-cell} ipython3
from pymatgen.core import Composition, Element

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

A spectrum is already a vector. Binning it to a fixed grid makes it a feature vector for classification (Chapter 39).

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

| Representation | Good for | Watch out |
|---|---|---|
| Descriptors | small data, interpretability | may miss local structure |
| Fingerprints | similarity, activity, toxicity | bit collisions, not invertible |
| Graphs | deep learning, large data | needs more data and compute |
| SMILES / SELFIES | language and generative models | many strings per molecule |
| Composition | inorganic materials | ignores structure and polymorphs |
| 3D / crystal graphs | quantum and materials properties | need geometries |
| Spectra | analytical chemistry | baseline, alignment, resolution |

## Exercises

**★ 1.** How many Morgan bits are set, on average, at radius 1, 2 and 3?

**★★ 2.** Find the 10 descriptors most correlated with `logS_sim`. Are any of them redundant?

**★★★ 3.** Add composition features to the binary-materials dataset and compare a random forest predicting `band_gap_sim_eV` with and without them.
