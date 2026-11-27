---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Catalysis: Descriptors, Volcano Plots and Screening

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 25, 26, 30 and Part V.
```

Catalysis is the art of lowering barriers. Computational catalysis found that whole families of reactions can be summarised by one or two **descriptors**, which produces the famous volcano plots {cite:p}`hammer1995`, and machine learning now predicts the adsorption energies those plots need, from datasets of millions of DFT calculations {cite:p}`chanussot2021`. A **catalyst** speeds up a reaction without being consumed, by offering a lower-energy path through a series of intermediates bound to its surface or metal centre. How strongly those intermediates bind decides how well the catalyst works, and computing binding energies with density functional theory (DFT) takes hours to days per surface. A **descriptor** is a single number, usually the binding energy of one key intermediate such as an oxygen atom, that captures most of the variation in activity across catalysts. Once activity is known as a function of the descriptor, the search for a better catalyst becomes a search for materials with the right descriptor value, and that value is much cheaper to predict. This chapter builds the chain step by step: scaling relations, the volcano plot, ML prediction of the descriptor, alloy screening, ligand maps for molecular catalysts, and a microkinetic model that links binding energy to turnover.

All energies in this chapter are illustrative values chosen to follow the known periodic trends, not results of real DFT calculations.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.core import Element
rng = np.random.default_rng(7)
```

## Scaling relations

On transition-metal surfaces the binding energies of related intermediates are linearly correlated: if you know how strongly a surface binds O, you can estimate how strongly it binds OH. That is why one descriptor can summarise a whole mechanism. Intermediates such as *OH and *OOH bind to the surface through their oxygen atom (the asterisk marks a surface-bound species), so a surface that holds O tightly also holds them tightly, but more weakly because the O is already partly bonded. A straight-line fit, `np.polyfit(x, y, 1)` (Chapter 16), gives the slope and intercept of such a **scaling relation**:

```{code-cell} ipython3
x_demo = np.array([-3.0, -2.0, -1.0])
y_demo = np.array([-1.6, -1.1, -0.6])
print(np.polyfit(x_demo, y_demo, 1))   # slope 0.5, intercept -0.1
```

The simulated surfaces below follow two such lines, with a little noise added.

```{code-cell} ipython3
metals = ["Ag", "Au", "Cu", "Pd", "Pt", "Ni", "Co", "Rh", "Ir", "Ru", "Fe", "Mn", "Re", "W", "Mo"]
# Illustrative oxygen binding energies (eV, more negative = stronger binding), ordered by periodic trend
dE_O = np.array([-0.4, 0.0, -1.0, -1.1, -1.2, -2.0, -2.4, -2.1, -1.9, -2.8, -3.1, -3.5, -3.6, -4.1, -3.9])
dE_OH = 0.5 * dE_O - 0.1 + rng.normal(0, 0.08, len(metals))     # a scaling relation with slope ≈ 0.5
dE_OOH = 0.5 * dE_O + 3.0 + rng.normal(0, 0.08, len(metals))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for y, label, c in [(dE_OH, "*OH", "#0f9d8a"), (dE_OOH, "*OOH", "#6c4ad6")]:
    ax.scatter(dE_O, y, s=70, color=c, edgecolor="k", label=label)
    k, b = np.polyfit(dE_O, y, 1)
    ax.plot(dE_O, k * dE_O + b, color=c, ls="--", lw=1)
    ax.text(-1.0, k * -1.0 + b + 0.25, f"slope {k:.2f}", color=c)
for x, y, m in zip(dE_O, dE_OH, metals):
    ax.annotate(m, (x, y), textcoords="offset points", xytext=(5, -10), fontsize=8)
ax.set(xlabel="ΔE(*O) / eV", ylabel="ΔE(*OH), ΔE(*OOH) / eV", title="Scaling relations on metal surfaces")
ax.legend(); plt.show()
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Because ΔE(*OOH) ≈ ΔE(*OH) + 3.2 eV on almost every metal, the four-electron oxygen reduction reaction has a built-in minimum overpotential of roughly 0.3–0.4 V. Breaking this scaling relation is one of the central goals of modern electrocatalysis.
```

## The volcano plot

Sabatier's principle says the best catalyst binds the intermediate **just right**: too weak and nothing reacts, too strong and the surface is poisoned. Plotting activity against the descriptor gives a volcano. On the strong-binding side, activity is limited by getting the product *off* the surface (desorption), and weakening the binding helps; on the weak-binding side, it is limited by activating the reactant *on* the surface (dissociation), and strengthening the binding helps. The overall rate is controlled by whichever step is slower, so activity is the **minimum** of two straight lines, which gives the volcano's peaked shape:

```{code-cell} ipython3
d_demo = np.array([-3.0, -2.0, -1.0, 0.0])
print(np.minimum(d_demo + 4, -d_demo + 1))   # rises, peaks, then falls
```

Placing each metal on the curve at its own descriptor value shows which metals are close to the top.

```{code-cell} ipython3
dE = np.linspace(-4.5, 0.5, 400)
# Two limiting steps: dissociation (easier for strong binders) and desorption (easier for weak binders)
rate_dissoc = 1.45 - 0.9 * dE        # weak binders: activating the reactant is slow
rate_desorb = 1.3 * (dE + 4.5)       # strong binders: the product sticks to the surface
activity = np.minimum(rate_dissoc, rate_desorb)

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(dE, rate_dissoc, ls=":", color="#e5487a", label="limited by dissociation")
ax.plot(dE, rate_desorb, ls=":", color="#3b82f6", label="limited by desorption")
ax.plot(dE, activity, color="k", lw=2.5)
ax.fill_between(dE, activity, activity.min(), color="#f2a93b", alpha=0.15)
metal_act = np.interp(dE_O, dE, activity)
ax.scatter(dE_O, metal_act, s=90, c=metal_act, cmap="plasma", edgecolor="k", zorder=5)
for x, y, m in zip(dE_O, metal_act, metals):
    ax.annotate(m, (x, y), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=9, weight="bold")
ax.axvline(dE[np.argmax(activity)], ls="--", color="grey")
ax.text(dE[np.argmax(activity)] + 0.1, activity.min() + 0.3, "Sabatier optimum", fontsize=9)
ax.set(xlabel="ΔE(*O) / eV  (descriptor)", ylabel="log(activity), arbitrary units", title="A volcano plot")
ax.legend(loc="lower left"); plt.show()
```

## Predicting adsorption energies with ML

Instead of a DFT calculation for every surface, learn ΔE from cheap features of the metal. This is the idea behind large catalysis datasets and models {cite:p}`chanussot2021`. Here each metal is described by properties that can be looked up instantly: atomic number, group and row in the periodic table, electronegativity, atomic radius, number of d electrons and melting point. pymatgen's `Element` class provides most of them:

```{code-cell} ipython3
pt = Element("Pt")
print(pt.Z, pt.group, pt.row, pt.X, pt.atomic_radius)
```

With only 15 metals, **leave-one-out** cross-validation is used: each metal in turn is predicted by a model trained on the other 14, which is the most data-efficient honest test for a very small dataset. A gradient-boosting regressor builds an ensemble of small trees, each correcting the errors of the previous ones.

```{code-cell} ipython3
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.metrics import mean_absolute_error

def metal_features(sym):
    e = Element(sym)
    return {"Z": e.Z, "group": e.group, "row": e.row, "X": e.X, "radius": float(e.atomic_radius),
            "n_d_electrons": {"Mn": 5, "Fe": 6, "Co": 7, "Ni": 8, "Cu": 10, "Mo": 5, "Ru": 7, "Rh": 8,
                              "Pd": 10, "Ag": 10, "W": 4, "Re": 5, "Ir": 7, "Pt": 9, "Au": 10}[sym],
            "melting_K": float(e.melting_point)}

F = pd.DataFrame([metal_features(m) for m in metals], index=metals)
model = GradientBoostingRegressor(random_state=0)
pred = cross_val_predict(model, F.values, dE_O, cv=LeaveOneOut())
print(f"Leave-one-metal-out MAE = {mean_absolute_error(dE_O, pred):.2f} eV")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
axes[0].scatter(dE_O, pred, s=80, color="#6c4ad6", edgecolor="k")
for x, y, m in zip(dE_O, pred, metals):
    axes[0].annotate(m, (x, y), textcoords="offset points", xytext=(6, -4), fontsize=8)
axes[0].plot([-4.5, 0.5], [-4.5, 0.5], "k--")
axes[0].set(xlabel="DFT-like ΔE(*O) / eV", ylabel="ML prediction", title="Learning the descriptor", aspect="equal")
model.fit(F.values, dE_O)
pd.Series(model.feature_importances_, index=F.columns).sort_values().plot.barh(ax=axes[1], color="#0f9d8a")
axes[1].set_title("Group (d-band filling) dominates")
plt.tight_layout(); plt.show()
```

The periodic-table group carries most of the importance, with the d-electron count second. Both measure how full the metal's d band is, which is the physical origin of the binding trend in the d-band model {cite:p}`hammer1995`: metals on the left of the transition series, with fewer d electrons, bind oxygen more strongly.

## Screening alloys

A simple interpolation model lets us screen binary alloys for descriptor values near the optimum, the classic computational-screening workflow. Alloying two metals gives a binding energy between theirs, so it is possible to "tune" the descriptor towards the top of the volcano. As a first approximation the alloy's descriptor is the composition-weighted average of the two metals' values, a linear interpolation (Vegard's law is the analogous rule for lattice constants), plus a small correction for their difference in electronegativity. For a 50:50 alloy of two metals:

```{code-cell} ipython3
x_frac, d_a, d_b = 0.5, -1.2, -0.4
print(x_frac * d_a + (1 - x_frac) * d_b, "eV")
```

`itertools.combinations` generates every pair of metals once, and three compositions of each are scored.

```{code-cell} ipython3
import itertools
opt = dE[np.argmax(activity)]
rows = []
for a, b in itertools.combinations(metals, 2):
    for x in (0.25, 0.5, 0.75):
        d = x * dE_O[metals.index(a)] + (1 - x) * dE_O[metals.index(b)]
        d += 0.25 * x * (1 - x) * (Element(a).X - Element(b).X)        # a small non-linear mixing term
        rows.append({"alloy": f"{a}{int(x*100)}{b}{int((1-x)*100)}", "dE_O": d, "activity": np.interp(d, dE, activity)})
alloys = pd.DataFrame(rows).sort_values("activity", ascending=False)
fig, ax = plt.subplots(figsize=(7.5, 4.5))
ax.scatter(alloys.dE_O, alloys.activity, s=12, color="lightgrey")
best = alloys.head(12)
ax.scatter(best.dE_O, best.activity, s=70, color="#e5487a", edgecolor="k")
for r in best.head(6).itertuples():
    ax.annotate(r.alloy, (r.dE_O, r.activity), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
ax.plot(dE, activity, color="k", lw=1.5)
ax.set(xlabel="ΔE(*O) / eV", ylabel="predicted activity", title=f"Screening {len(alloys)} alloy compositions")
plt.show()
best.head(8).round(2)
```

```{warning}
Interpolating descriptors across compositions ignores segregation, ordering, oxide formation and the real surface under reaction conditions. Treat screening hits as hypotheses for DFT and experiment, never as results.
```

## Homogeneous catalysis: ligand design

For molecular catalysts the descriptors are ligand properties: electron richness (Tolman electronic parameter) and steric bulk (cone angle, buried volume). Our simulated C–N coupling screen from Chapter 37 shows exactly that two-dimensional structure. In homogeneous catalysis the catalyst is a metal complex dissolved in the reaction mixture, and its activity is tuned by the ligands bound to the metal. Electron-rich ligands push electron density onto the metal and speed up steps such as oxidative addition; bulky ligands crowd the metal and speed up steps such as reductive elimination, but too much bulk blocks the substrate. A **ligand map** plots each ligand by its steric and electronic parameters, so a good region of the map suggests which new ligands to try. Grouping the screen by the two ligand parameters with `groupby` and averaging the yield gives one point per ligand.

```{code-cell} ipython3
from pathlib import Path
DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name)

hte = load_data("cn_coupling_hte_sim.csv")
grid = hte.groupby(["ligand_electron_richness", "ligand_cone_angle"]).yield_sim.mean().reset_index()
fig, ax = plt.subplots(figsize=(6.5, 4.5))
sc = ax.scatter(grid.ligand_cone_angle, grid.ligand_electron_richness, c=grid.yield_sim, s=700, cmap="RdYlGn",
                edgecolor="k", vmin=30, vmax=80)
for r in grid.itertuples():
    ax.annotate(f"{r.yield_sim:.0f} %", (r.ligand_cone_angle, r.ligand_electron_richness), ha="center", va="center", fontsize=9)
ax.set(xlabel="cone angle / °  (sterics)", ylabel="electron richness  (electronics)",
       title="Ligand map: a steric–electronic sweet spot")
plt.colorbar(sc, label="mean yield / %"); plt.show()
```

## Microkinetics: from barriers to turnover

A descriptor is only useful because it controls the rate. Here is a two-step catalytic cycle solved as a steady-state microkinetic model. A **microkinetic model** writes a rate equation for every elementary step and solves them together (Chapter 25). In this simple Langmuir–Hinshelwood cycle the reactant first adsorbs, reaching equilibrium with a surface **coverage** θ, the fraction of sites occupied, given by the Langmuir isotherm (Chapter 26): θ = Kp/(1 + Kp), where the equilibrium constant K grows as the adsorption becomes more exothermic. Coverage alone rises steadily with binding strength:

```{code-cell} ipython3
for K in [0.1, 1, 10, 100]:
    print(K, round(K / (1 + K), 3))
```

The surface reaction then needs an occupied site next to a free one, so the rate is proportional to θ(1 − θ), and its barrier grows when the intermediate is bound too strongly. The **turnover frequency** (TOF), the number of reaction events per site per second, therefore passes through a maximum, a volcano that emerges from the kinetics rather than being drawn by hand.

```{code-cell} ipython3
from scipy.optimize import brentq
kB, T = 8.617e-5, 500.0        # eV/K, K

def turnover(dE_ads, Ea_surface=0.9, p=1.0):
    """Simple Langmuir-Hinshelwood cycle: adsorption equilibrium then surface reaction."""
    K_ads = np.exp(-dE_ads / (kB * T))
    theta = K_ads * p / (1 + K_ads * p)
    k_surf = 1e13 * np.exp(-(Ea_surface + 0.5 * max(0.0, -dE_ads - 1.0)) / (kB * T))
    return k_surf * theta * (1 - theta)

ads = np.linspace(-2.5, 0.2, 300)
tof = [turnover(d) for d in ads]
plt.semilogy(ads, tof, color="#6c4ad6", lw=2.5)
plt.axvline(ads[int(np.argmax(tof))], ls="--", color="grey")
plt.xlabel("ΔE(adsorption) / eV"); plt.ylabel("turnover frequency / s$^{-1}$")
plt.title("A volcano emerging from a microkinetic model"); plt.show()
```

```{admonition} Summary
:class: tip
- A descriptor, such as ΔE(*O), summarises catalytic activity across many materials.
- Scaling relations link the binding energies of related intermediates through straight lines.
- The volcano plot follows from Sabatier's principle: activity is the minimum of two limiting lines.
- ML predicts descriptors from cheap elemental features; use leave-one-out for tiny datasets.
- Alloy screening interpolates descriptors; treat hits as hypotheses.
- Ligand maps use steric and electronic parameters; microkinetic models turn binding energies into turnover frequencies.
```

## Exercises

**★ 1.** Shift the scaling relation for *OOH down by 0.3 eV and recompute the volcano. How much does peak activity improve?

**★★ 2.** Add a steric term to the ligand map and find the optimum cone angle for each electron richness.

**★★★ 3.** Extend the microkinetic model to three steps with explicit coverages of two intermediates, and solve the steady state with `scipy.optimize.fsolve`.
