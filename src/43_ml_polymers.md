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

Polymers break the assumptions of molecular cheminformatics: a sample is a *distribution* of chain lengths, not one molecule, and the same repeat unit can give very different materials. Polymer informatics therefore needs its own notation {cite:p}`lin2019` and its own feature design {cite:p}`kim2018`. A **polymer** is a long chain built from many copies of a small **repeat unit**: polyethylene from –CH₂–CH₂–, polystyrene from –CH₂–CH(C₆H₅)–. Its useful properties, such as stiffness, toughness, solubility and the temperature at which it softens, depend on the chemistry of the repeat unit, on how long the chains are and how their lengths are distributed, and on how chains pack together. The property used throughout this chapter is the **glass-transition temperature** Tg: below Tg the chains are frozen in place and the material is a rigid glass; above it, segments of the chain can move and the material becomes rubbery or soft. Tg decides whether a plastic is a rigid cup or a flexible film at room temperature, so predicting it from structure is a classic polymer-informatics task. The chapter covers repeat-unit notation, building oligomers, polymer features, a Tg model, molar-mass distributions and copolymer design.

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

A repeat unit is written with two attachment points. RDKit reads `*` as a dummy atom. The two stars mark where the unit bonds to the previous and the next unit in the chain, so `*CC*` is polyethylene and `*CC(C)*` is polypropylene. A dummy atom has atomic number 0, which makes it easy to exclude when counting real atoms:

```{code-cell} ipython3
m_pp = Chem.MolFromSmiles("[*]CC(C)[*]")
print([(a.GetSymbol(), a.GetAtomicNum()) for a in m_pp.GetAtoms()])
```

```{code-cell} ipython3
def repeat_mol(smi):
    return Chem.MolFromSmiles(smi.replace("*", "[*]"))

named = poly[poly.source == "approximate textbook value"]
Draw.MolsToGridImage([repeat_mol(s) for s in named.repeat_unit[:12]], legends=list(named.name[:12]),
                     molsPerRow=4, subImgSize=(200, 150))
```

**BigSMILES** extends this to real architectures, writing a copolymer as `{[>]CC(c1ccccc1)[<], [>]CC[<]}` with explicit bonding descriptors, so that random, block and graft structures can be distinguished.

## Building an oligomer

To compute conformational or 3D properties you need an actual chain. Here we stitch *n* repeat units together. Because a SMILES string is read left to right, deleting the two stars and writing the remaining text *n* times links the end of each unit to the start of the next. For polyethylene's `CC`, three copies give hexane:

```{code-cell} ipython3
print("CC" * 3, Chem.MolToSmiles(Chem.MolFromSmiles("CC" * 3)))
```

This trick works when the first atom written is one attachment point and the last atom is the other, which is how the repeat units in this dataset are written. The ends of the oligomer are capped with hydrogens automatically.

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

Features are computed on the repeat unit, and chosen to reflect what controls chain mobility. Aromatic and aliphatic rings make the backbone stiff; rotatable bonds make it flexible; polar groups (measured by TPSA and hydrogen-bond donors and acceptors, Chapter 20) increase the attraction between chains; and bulky groups, such as a carbon with no hydrogens carrying two methyl groups (the *quaternary* carbon of PMMA), hinder rotation. SMARTS patterns (Chapter 21) flag esters, amides and ethers. Atoms with two or more neighbours are counted as a rough measure of the backbone, and the remaining atoms as side chain.

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

The dataset is small, so **leave-one-out** cross-validation is used (Chapter 38): every polymer is predicted by a model trained on all the others. Two models are compared, ridge regression, which assumes each feature adds a fixed amount to Tg, and gradient boosting, which can capture interactions and thresholds.

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

Both models are only moderately accurate, with errors of several tens of degrees, and the simpler ridge model does better than gradient boosting. That is common with small datasets, where a flexible model has too few examples to learn interactions reliably; the plot shows the gradient-boosting predictions, which the next section interprets.

## What raises the glass transition?

Feature importances say which features the model uses, but not in which direction. A **partial-dependence plot** answers that: it sets one feature to each value in turn for every polymer in the dataset, averages the model's predictions, and plots that average against the feature value. A rising curve means that increasing the feature raises the predicted Tg, all else being equal.

```{code-cell} ipython3
gb = GradientBoostingRegressor(random_state=0).fit(P.values, y)
imp = pd.Series(gb.feature_importances_, index=P.columns).sort_values()
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
imp.tail(8).plot.barh(ax=axes[0], color="#0f9d8a"); axes[0].set_title("Feature importance for Tg")
from sklearn.inspection import PartialDependenceDisplay
PartialDependenceDisplay.from_estimator(gb, P, ["aromatic_rings", "rot_bonds"], ax=axes[1], line_kw={"color": "#e5487a"})
plt.tight_layout(); plt.show()
```

The partial-dependence curves follow the textbook picture: stiff aromatic rings push Tg up and flexible rotatable bonds pull it down. Polarity (TPSA) is the single most important feature, which also fits, because polar groups attract neighbouring chains. Not everything the model learned is chemistry, though. In this small dataset the model also predicts a higher Tg for larger side chains and a slightly lower one for quaternary carbons, the opposite of the textbook effects of long flexible side chains and bulky groups; with fewer than a hundred polymers, such patterns can come from a handful of examples. Partial dependence describes the model, not nature.

```{admonition} ⚗️ Chemistry note
:class: chem-note
Tg is not a thermodynamic transition. It depends on cooling rate, molar mass (via the Fox–Flory relation $T_g = T_g^\infty - K/M_n$), tacticity and plasticiser content. A model trained on single numbers can never capture that, so always state the measurement conditions with the data.
```

## Molar-mass distributions

A polymer sample is a distribution. Two averages matter: the number average $M_n$ and the weight average $M_w$, whose ratio is the **polydispersity index**. The number average $M_n = \sum N_i M_i / \sum N_i$ is the ordinary mean mass per chain, where $N_i$ chains have mass $M_i$. The weight average $M_w = \sum N_i M_i^2 / \sum N_i M_i$ weights each chain by its own mass, so long chains count more; it is always at least $M_n$. The polydispersity index (PDI), $M_w/M_n$, is 1 when all chains are the same length and grows as the distribution widens. A sample of three chains makes the difference clear:

```{code-cell} ipython3
masses = np.array([10_000, 20_000, 90_000])
Mn_demo = masses.mean()
Mw_demo = (masses ** 2).sum() / masses.sum()
print(f"Mn = {Mn_demo:.0f}, Mw = {Mw_demo:.0f}, PDI = {Mw_demo / Mn_demo:.2f}")
```

In step-growth polymerisation, such as the formation of polyesters, the chain-length distribution follows the **Flory–Schulz** form, controlled by the conversion $p$, the fraction of functional groups that have reacted. High molar mass needs conversions very close to 1, and the PDI approaches 2. Experimentally, the distribution is measured by **gel permeation chromatography** (GPC), which separates chains by size; a log-normal peak in molar mass is a simple model of the trace, with the integrals for $M_n$ and $M_w$ done by `np.trapezoid`.

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

Here $w_1$ and $w_2$ are the weight fractions of the two monomers and the temperatures are in kelvin. A **random copolymer** contains both monomers mixed along each chain, and the Fox equation averages the *reciprocals* of their Tg values, so it always lies below a straight-line average. A 50:50 copolymer of two polymers with Tg = 373 K and 223 K:

```{code-cell} ipython3
T_fox = 1 / (0.5 / 373 + 0.5 / 223)
print(f"Fox {T_fox - 273.15:.1f} °C, linear average {(373 + 223) / 2 - 273.15:.1f} °C")
```

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

The Fox equation can also be run backwards. Solving it for $w_1$ gives the composition that produces a chosen Tg, $w_1 = (1/T_g - 1/T_{g,2})/(1/T_{g,1} - 1/T_{g,2})$, which has a solution only when the target lies between the two homopolymer values. Looping over all pairs of named polymers lists every copolymer that can reach 60 °C, and sorting by how close $w_A$ is to 0.5 puts the most balanced compositions first.

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

```{admonition} Summary
:class: tip
- Repeat units are written with two `*` attachment points; BigSMILES describes copolymer architectures.
- Oligomers can be built by repeating the repeat-unit SMILES.
- Features reflecting stiffness, flexibility, polarity and bulk predict Tg; partial-dependence plots show the direction of each effect.
- A polymer sample is a distribution: $M_n$, $M_w$ and PDI summarise it.
- The Fox equation predicts random-copolymer Tg and can be inverted to design a composition.
```

## Exercises

**★ 1.** Which repeat unit in the dataset has the highest predicted Tg? Does the structure explain it?

**★★ 2.** Add a "backbone aromaticity fraction" feature and see whether the leave-one-out error improves.

**★★★ 3.** Implement the Fox–Flory molar-mass correction and use it to predict Tg for polystyrene at Mn = 2 000, 10 000 and 100 000 g/mol.
