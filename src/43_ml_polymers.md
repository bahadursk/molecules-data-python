---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Polymers and Soft Matter

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 20, 31 and 32.
```

Polymers break the assumptions of molecular cheminformatics: a sample is a *distribution* of chain lengths, not one molecule, and the same repeat unit can give very different materials. Polymer informatics therefore needs its own notation {cite:p}`lin2019` and its own feature design {cite:p}`kim2018`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, rdMolDescriptors, rdFingerprintGenerator

DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name, **kw):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name, **kw)

poly = load_data("polymers_sim.csv")
poly.head()
```

```{admonition} About the data
:class: warning
The 20 named polymers carry **approximate textbook** glass-transition temperatures; the remaining rows are **simulated** from a linear model fitted to those, plus noise. Real polymer informatics uses measured databases such as PoLyInfo or Polymer Genome {cite:p}`kim2018`.
```

## Representing a repeat unit

A repeat unit is written with two attachment points. RDKit reads `*` as a dummy atom.

```{code-cell} ipython3
def repeat_mol(smi):
    return Chem.MolFromSmiles(smi.replace("*", "[*]"))

named = poly[poly.source == "approximate textbook value"]
Draw.MolsToGridImage([repeat_mol(s) for s in named.repeat_unit[:12]], legends=list(named.name[:12]),
                     molsPerRow=4, subImgSize=(200, 150))
```

**BigSMILES** extends this to real architectures, writing a copolymer as `{[>]CC(c1ccccc1)[<], [>]CC[<]}` with explicit bonding descriptors, so that random, block and graft structures can be distinguished.

## Building an oligomer

To compute conformational or 3D properties you need an actual chain. Here we stitch *n* repeat units together.

```{code-cell} ipython3
def build_oligomer(unit_smi, n=6):
    """Join n copies of a repeat unit written with two * attachment points."""
    body = unit_smi.replace("*", "")
    return Chem.MolFromSmiles(body * n)

ps = build_oligomer("*CC(c1ccccc1)*", 5)
print("polystyrene pentamer:", Chem.MolToSmiles(ps), "| MW =", round(Descriptors.MolWt(ps), 1))
Draw.MolToImage(ps, size=(520, 220))
```

## Features for polymers

```{code-cell} ipython3
def polymer_features(smi):
    m = repeat_mol(smi)
    heavy = [a for a in m.GetAtoms() if a.GetAtomicNum() > 0]
    backbone_len = len([a for a in heavy if a.GetDegree() >= 2])
    return {"n_heavy": len(heavy), "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(m),
            "aliphatic_rings": rdMolDescriptors.CalcNumAliphaticRings(m),
            "rot_bonds": rdMolDescriptors.CalcNumRotatableBonds(m), "tpsa": rdMolDescriptors.CalcTPSA(m),
            "logp": Descriptors.MolLogP(m), "hbd": rdMolDescriptors.CalcNumHBD(m),
            "hba": rdMolDescriptors.CalcNumHBA(m), "fsp3": rdMolDescriptors.CalcFractionCSP3(m),
            "n_F": sum(a.GetSymbol() == "F" for a in heavy), "n_Cl": sum(a.GetSymbol() == "Cl" for a in heavy),
            "quaternary": int(m.HasSubstructMatch(Chem.MolFromSmarts("[CX4;H0](C)(C)"))),
            "has_ester": int(m.HasSubstructMatch(Chem.MolFromSmarts("C(=O)O[CX4]"))),
            "has_amide": int(m.HasSubstructMatch(Chem.MolFromSmarts("C(=O)[NX3]"))),
            "has_ether": int(m.HasSubstructMatch(Chem.MolFromSmarts("[CX4][OX2][CX4]"))),
            "backbone_atoms": backbone_len, "side_chain_atoms": len(heavy) - backbone_len}

P = pd.DataFrame([polymer_features(s) for s in poly.repeat_unit]).astype(float)
P.head(3)
```

## Predicting the glass transition

```{code-cell} ipython3
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, LeaveOneOut, KFold
from sklearn.metrics import mean_absolute_error, r2_score

y = poly.Tg_C.values
loo = LeaveOneOut()
for name, model in [("ridge", RidgeCV(alphas=np.logspace(-2, 3, 20))),
                    ("gradient boosting", GradientBoostingRegressor(random_state=0))]:
    p = cross_val_predict(model, P.values, y, cv=loo)
    print(f"{name:20s} leave-one-out MAE = {mean_absolute_error(y, p):5.1f} °C,  R² = {r2_score(y, p):.2f}")

p = cross_val_predict(GradientBoostingRegressor(random_state=0), P.values, y, cv=loo)
fig, ax = plt.subplots(figsize=(6, 5.5))
is_named = (poly.source == "approximate textbook value").values
ax.scatter(y[is_named], p[is_named], s=70, color="#6c4ad6", edgecolor="k", label="named polymers")
ax.scatter(y[~is_named], p[~is_named], s=30, color="#f2a93b", label="simulated")
for i in np.where(is_named)[0]:
    ax.annotate(poly.name[i], (y[i], p[i]), textcoords="offset points", xytext=(5, 4), fontsize=7)
ax.plot([-130, 170], [-130, 170], "k--"); ax.set(xlabel="Tg / °C", ylabel="predicted Tg / °C", aspect="equal",
                                                 title="Leave-one-out prediction of Tg")
ax.legend(); plt.show()
```

## What raises the glass transition?

```{code-cell} ipython3
gb = GradientBoostingRegressor(random_state=0).fit(P.values, y)
imp = pd.Series(gb.feature_importances_, index=P.columns).sort_values()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
imp.tail(8).plot.barh(ax=axes[0], color="#0f9d8a"); axes[0].set_title("Feature importance for Tg")
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(gb, P, ["aromatic_rings", "rot_bonds"], ax=axes[1], line_kw={"color": "#e5487a"})
plt.tight_layout(); plt.show()
```

Stiff aromatic rings and bulky quaternary carbons push Tg up; flexible rotatable bonds and long side chains push it down. That is the textbook picture, recovered from data.

```{admonition} ⚗️ Chemistry note
:class: chem-note
Tg is not a thermodynamic transition. It depends on cooling rate, molar mass (via the Fox–Flory relation $T_g = T_g^\infty - K/M_n$), tacticity and plasticiser content. A model trained on single numbers can never capture that, so always state the measurement conditions with the data.
```

## Molar-mass distributions

A polymer sample is a distribution. Two averages matter: the number average $M_n$ and the weight average $M_w$, whose ratio is the **polydispersity index**.

```{code-cell} ipython3
def flory_schulz(p, n_max=4000):
    """Chain-length distribution for step-growth polymerisation with conversion p."""
    n = np.arange(1, n_max)
    w = (1 - p) ** 2 * n * p ** (n - 1)
    return n, w / w.sum()

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for p_conv, c in zip([0.95, 0.98, 0.995], ["#3b82f6", "#f2a93b", "#e5487a"]):
    n, w = flory_schulz(p_conv)
    Mn = 1 / (1 - p_conv)          # number-average degree of polymerisation
    Mw = (1 + p_conv) / (1 - p_conv)
    axes[0].plot(n[:600], w[:600], color=c, label=f"p = {p_conv}, PDI = {Mw / Mn:.2f}")
axes[0].set(xlabel="degree of polymerisation", ylabel="weight fraction", title="Flory–Schulz distributions")
axes[0].legend()

# A simulated GPC trace: log-normal distribution in molar mass
M = np.logspace(3, 6, 400)
for sigma, c in zip([0.2, 0.45, 0.8], ["#0f9d8a", "#6c4ad6", "#e5487a"]):
    w = np.exp(-0.5 * ((np.log(M) - np.log(5e4)) / sigma) ** 2)
    Mn = np.trapezoid(w, M) / np.trapezoid(w / M, M)
    Mw = np.trapezoid(w * M, M) / np.trapezoid(w, M)
    axes[1].semilogx(M, w / w.max(), color=c, label=f"PDI = {Mw / Mn:.2f}")
axes[1].set(xlabel="molar mass / g mol$^{-1}$", ylabel="normalised signal", title="Simulated GPC traces"); axes[1].legend()
plt.tight_layout(); plt.show()
```

## Copolymer design

For a random copolymer, Tg often follows the Fox equation:

$$\frac{1}{T_g} = \frac{w_1}{T_{g,1}} + \frac{w_2}{T_{g,2}}$$

```{code-cell} ipython3
pairs = [("polystyrene", "polybutadiene"), ("PMMA", "poly(vinyl acetate)"), ("PVC", "PMMA")]
w = np.linspace(0, 1, 100)
fig, ax = plt.subplots(figsize=(7, 4.5))
tg_lookup = dict(zip(poly.name, poly.Tg_C))
for (a, b), c in zip(pairs, ["#6c4ad6", "#0f9d8a", "#e5487a"]):
    T1, T2 = tg_lookup[a] + 273.15, tg_lookup[b] + 273.15
    fox = 1 / (w / T1 + (1 - w) / T2) - 273.15
    linear = w * (T1 - 273.15) + (1 - w) * (T2 - 273.15)
    ax.plot(w, fox, color=c, lw=2.5, label=f"{a} / {b} (Fox)")
    ax.plot(w, linear, color=c, ls=":", lw=1.2)
ax.set(xlabel="weight fraction of the first component", ylabel="Tg / °C",
       title="Fox equation (solid) vs linear mixing (dotted)")
ax.legend(fontsize=8); plt.show()
```

### Designing to a target

```{code-cell} ipython3
target = 60.0
rows = []
for a in poly.name[:20]:
    for b in poly.name[:20]:
        if a >= b:
            continue
        T1, T2 = tg_lookup[a] + 273.15, tg_lookup[b] + 273.15
        if min(T1, T2) - 273.15 < target < max(T1, T2) - 273.15:
            wa = (1 / (target + 273.15) - 1 / T2) / (1 / T1 - 1 / T2)
            rows.append({"A": a, "B": b, "w_A": wa, "Tg_A": T1 - 273.15, "Tg_B": T2 - 273.15})
blends = pd.DataFrame(rows)
print(f"{len(blends)} binary copolymers can hit Tg = {target:.0f} °C")
blends["balance"] = (blends.w_A - 0.5).abs()
blends.sort_values("balance").head(8).round(2)
```

## Exercises

**★ 1.** Which repeat unit in the dataset has the highest predicted Tg? Does the structure explain it?

**★★ 2.** Add a "backbone aromaticity fraction" feature and see whether the leave-one-out error improves.

**★★★ 3.** Implement the Fox–Flory molar-mass correction and use it to predict Tg for polystyrene at Mn = 2 000, 10 000 and 100 000 g/mol.
