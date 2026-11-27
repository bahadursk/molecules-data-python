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

Catalysis is the art of lowering barriers. Computational catalysis found that whole families of reactions can be summarised by one or two **descriptors**, which produces the famous volcano plots {cite:p}`hammer1995`, and machine learning now predicts the adsorption energies those plots need, from datasets of millions of DFT calculations {cite:p}`chanussot2021`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymatgen.core import Element
rng = np.random.default_rng(7)
```

## Scaling relations

On transition-metal surfaces the binding energies of related intermediates are linearly correlated: if you know how strongly a surface binds O, you can estimate how strongly it binds OH. That is why one descriptor can summarise a whole mechanism.

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

Sabatier's principle says the best catalyst binds the intermediate **just right**: too weak and nothing reacts, too strong and the surface is poisoned. Plotting activity against the descriptor gives a volcano.

```{code-cell} ipython3
dE = np.linspace(-4.5, 0.5, 400)
# Two limiting steps: dissociation (easier for strong binders) and desorption (easier for weak binders)
rate_dissoc = -(dE + 4.5) * 0.9
rate_desorb = (dE + 1.0) * 1.3 + 3.2
activity = np.minimum(rate_dissoc + 3.5, -rate_desorb + 5.2)

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(dE, rate_dissoc + 3.5, ls=":", color="#e5487a", label="limited by dissociation")
ax.plot(dE, -rate_desorb + 5.2, ls=":", color="#3b82f6", label="limited by desorption")
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

Instead of a DFT calculation for every surface, learn ΔE from cheap features of the metal. This is the idea behind large catalysis datasets and models {cite:p}`chanussot2021`.

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
axes[1].set_title("The d-electron count dominates")
plt.tight_layout(); plt.show()
```

## Screening alloys

A simple interpolation model lets us screen binary alloys for descriptor values near the optimum, the classic computational-screening workflow.

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

For molecular catalysts the descriptors are ligand properties: electron richness (Tolman electronic parameter) and steric bulk (cone angle, buried volume). Our simulated C–N coupling screen from Chapter 37 shows exactly that two-dimensional structure.

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

A descriptor is only useful because it controls the rate. Here is a two-step catalytic cycle solved as a steady-state microkinetic model.

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

## Exercises

**★ 1.** Shift the scaling relation for *OOH down by 0.3 eV and recompute the volcano. How much does peak activity improve?

**★★ 2.** Add a steric term to the ligand map and find the optimum cone angle for each electron richness.

**★★★ 3.** Extend the microkinetic model to three steps with explicit coverages of two intermediates, and solve the steady state with `scipy.optimize.fsolve`.
