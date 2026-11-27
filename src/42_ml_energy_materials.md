---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Energy Materials: Batteries, Photovoltaics and MOFs

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 30, 41 and Part V.
```

Energy technologies are limited by materials: how much charge a battery electrode can store and how long it lasts, how efficiently a solar absorber converts light, how much gas a porous framework can hold. Each has become a machine-learning problem, from battery lifetime prediction {cite:p}`severson2019` to the diversity of the MOF universe {cite:p}`moosavi2020`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.constants import e as q_e, N_A
rng = np.random.default_rng(5)
```

## Batteries: theoretical capacity from first principles

The specific capacity of an electrode material follows directly from stoichiometry:

$$Q = \frac{n F}{3.6\,M} \quad [\mathrm{mAh\,g^{-1}}]$$

```{code-cell} ipython3
from pymatgen.core import Composition
F_const = 96485.3

CATHODES = {"LiCoO2": 1, "LiFePO4": 1, "LiMn2O4": 1, "LiNi0.8Co0.15Al0.05O2": 1, "Li2MnO3": 2,
            "LiNi0.33Mn0.33Co0.33O2": 1, "NaFePO4": 1, "Li2FeSiO4": 2, "LiVPO4F": 1, "S8": 16}
rows = []
for formula, n in CATHODES.items():
    M = Composition(formula).weight
    rows.append({"material": formula, "n_e": n, "M": float(M), "capacity_mAh_g": n * F_const / (3.6 * float(M))})
cap = pd.DataFrame(rows).sort_values("capacity_mAh_g")
fig, ax = plt.subplots(figsize=(7.5, 4))
ax.barh(cap.material, cap.capacity_mAh_g, color=plt.cm.viridis(cap.capacity_mAh_g / cap.capacity_mAh_g.max()))
for r in cap.itertuples():
    ax.text(r.capacity_mAh_g + 8, r.material, f"{r.capacity_mAh_g:.0f}", va="center", fontsize=9)
ax.set(xlabel="theoretical capacity / mAh g$^{-1}$", title="Why sulfur cathodes are so attractive")
plt.tight_layout(); plt.show()
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Theoretical capacity is an upper bound. Practical capacity is limited by how much lithium can be removed before the structure collapses, by kinetics and by side reactions. For LiCoO₂ only about half the lithium is usable.
```

## Energy density and the Ragone picture

```{code-cell} ipython3
chem = pd.DataFrame({
    "chemistry": ["Pb-acid", "Ni-MH", "LFP", "NMC-622", "NCA", "Li-S", "Li-air (theor.)", "Na-ion"],
    "voltage_V": [2.0, 1.2, 3.3, 3.7, 3.7, 2.2, 2.9, 3.1],
    "capacity_mAh_g": [120, 300, 160, 180, 200, 1200, 3800, 120],
    "cycle_life": [500, 1000, 5000, 1500, 1000, 200, 50, 3000]})
chem["Wh_kg_cell"] = chem.voltage_V * chem.capacity_mAh_g / 1000 * 0.4      # ≈40 % of electrode-level value at cell level
fig, ax = plt.subplots(figsize=(7.5, 5))
sc = ax.scatter(chem.Wh_kg_cell, chem.cycle_life, s=chem.voltage_V * 60, c=chem.capacity_mAh_g, cmap="plasma",
                edgecolor="k")
for r in chem.itertuples():
    ax.annotate(r.chemistry, (r.Wh_kg_cell, r.cycle_life), textcoords="offset points", xytext=(8, 5), fontsize=9)
ax.set(xscale="log", yscale="log", xlabel="estimated cell energy density / Wh kg$^{-1}$", ylabel="cycle life",
       title="The battery trade-off (bubble size = voltage)")
plt.colorbar(sc, label="electrode capacity / mAh g$^{-1}$"); plt.show()
```

## Predicting battery lifetime from early cycles

Severson and co-workers showed that the **change in discharge-capacity curves over the first 100 cycles** predicts total lifetime, long before any capacity fade is visible {cite:p}`severson2019`. Here we simulate cells with that structure and reproduce the workflow.

```{code-cell} ipython3
def simulate_cell(rng):
    """A cell with a hidden degradation rate; early-cycle curve changes leak information about it."""
    k_sei = rng.lognormal(-7.0, 0.5)               # SEI growth rate
    k_lam = rng.lognormal(-9.0, 0.7)               # loss of active material
    n = np.arange(1, 1001)
    cap = 1.0 - k_sei * np.sqrt(n) - k_lam * n
    cap += rng.normal(0, 0.0015, n.size)
    eol = np.argmax(cap < 0.8) + 1 if (cap < 0.8).any() else 1000
    ir = 0.02 + 0.5 * k_sei * np.sqrt(n) + rng.normal(0, 2e-4, n.size)     # internal resistance
    return cap, ir, eol

cells = [simulate_cell(np.random.default_rng(i)) for i in range(160)]
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for cap, ir, eol in cells[:30]:
    axes[0].plot(cap[:600], lw=0.8, color=plt.cm.coolwarm(min(eol, 1000) / 1000))
axes[0].axhline(0.8, ls="--", color="k"); axes[0].set(xlabel="cycle", ylabel="relative capacity", title="Simulated cycling")
axes[1].hist([c[2] for c in cells], bins=25, color="#6c4ad6")
axes[1].set(xlabel="cycle life (80 % capacity)", title="Lifetime distribution")
plt.tight_layout(); plt.show()
```

```{code-cell} ipython3
from sklearn.linear_model import ElasticNetCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_percentage_error

def early_features(cap, ir, window=100):
    d = cap[:window]
    return {"cap_10": d[9], "cap_100": d[-1], "delta_cap": d[-1] - d[9],
            "slope": np.polyfit(np.arange(window), d, 1)[0], "curvature": np.polyfit(np.arange(window), d, 2)[0],
            "var_delta": np.var(np.diff(d)), "ir_100": ir[window - 1], "d_ir": ir[window - 1] - ir[9],
            "log_abs_delta": np.log10(abs(d[-1] - d[9]) + 1e-9)}

Xb = pd.DataFrame([early_features(c, i) for c, i, _ in cells])
yb = np.log10([e for *_, e in cells])
model = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[0.1, 0.5, 0.9], cv=5, max_iter=5000))
pred = cross_val_predict(model, Xb.values, yb, cv=KFold(5, shuffle=True, random_state=0))
print(f"MAPE on cycle life = {mean_absolute_percentage_error(10 ** yb, 10 ** pred):.1%}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
axes[0].scatter(10 ** yb, 10 ** pred, s=30, color="#0f9d8a", edgecolor="k")
axes[0].plot([100, 1000], [100, 1000], "k--"); axes[0].set(xscale="log", yscale="log", xlabel="true cycle life",
                                                           ylabel="predicted from first 100 cycles", aspect="equal",
                                                           title="Lifetime prediction")
model.fit(Xb.values, yb)
coefs = pd.Series(model[-1].coef_, index=Xb.columns).sort_values()
coefs.plot.barh(ax=axes[1], color=["#e5487a" if c < 0 else "#0f9d8a" for c in coefs])
axes[1].set_title("Which early signals matter")
plt.tight_layout(); plt.show()
```

The strongest predictor is the *change* in the capacity curve, not its absolute value, exactly the finding that made this approach famous.

## Photovoltaics: the Shockley–Queisser limit

The band gap of an absorber sets a hard ceiling on efficiency: too small and photons are wasted as heat, too large and most of the spectrum is not absorbed.

```{code-cell} ipython3
h, c, kB = 6.62607015e-34, 2.99792458e8, 1.380649e-23

def blackbody_photon_flux(lam, T=5800):
    return 2 * np.pi * c / lam ** 4 / (np.exp(h * c / (lam * kB * T)) - 1)

lam = np.linspace(200e-9, 4000e-9, 4000)
flux = blackbody_photon_flux(lam) * 2.17e-5                  # scaled to ≈1000 W/m² total
energy = h * c / lam / q_e

def sq_efficiency(Eg):
    absorbed = energy >= Eg
    j = np.trapezoid(flux[absorbed][::-1], lam[absorbed][::-1])          # photons above the gap
    p_in = np.trapezoid((flux * energy)[::-1], lam[::-1])
    V_oc = 0.75 * Eg                                                      # empirical thermodynamic loss
    return j * V_oc * 0.85 / p_in

gaps = np.linspace(0.4, 3.0, 120)
eff = np.array([sq_efficiency(g) for g in gaps])
materials = {"Si": 1.12, "GaAs": 1.42, "CdTe": 1.50, "CIGS": 1.15, "MAPbI₃": 1.55, "a-Si": 1.70,
             "Ge": 0.67, "InP": 1.34, "CsPbBr₃": 2.30, "ZnO": 3.37}
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(gaps, 100 * eff, color="k", lw=2.5)
ax.fill_between(gaps, 100 * eff, color="#f2a93b", alpha=0.2)
for name, g in materials.items():
    if g <= 3.0:
        ax.scatter(g, 100 * sq_efficiency(g), s=90, edgecolor="k", zorder=5,
                   color="#0f9d8a" if 1.1 <= g <= 1.6 else "#e5487a")
        ax.annotate(name, (g, 100 * sq_efficiency(g)), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)
ax.axvspan(1.1, 1.6, color="#0f9d8a", alpha=0.08)
ax.set(xlabel="band gap / eV", ylabel="limiting efficiency / %", title="Detailed-balance limit and real absorbers")
plt.show()
```

### Screening perovskite compositions

```{code-cell} ipython3
A_ions = {"Cs": 1.88, "MA": 2.17, "FA": 2.53, "Rb": 1.72}
B_ions = {"Pb": 1.19, "Sn": 1.10, "Ge": 0.73}
X_ions = {"I": 2.20, "Br": 1.96, "Cl": 1.81}
GAP0 = {("Pb", "I"): 1.55, ("Pb", "Br"): 2.30, ("Pb", "Cl"): 3.00,
        ("Sn", "I"): 1.30, ("Sn", "Br"): 2.05, ("Sn", "Cl"): 2.80,
        ("Ge", "I"): 1.60, ("Ge", "Br"): 2.40, ("Ge", "Cl"): 3.20}
rows = []
for A, rA in A_ions.items():
    for B, rB in B_ions.items():
        for X, rX in X_ions.items():
            for frac in np.linspace(0, 1, 5):                    # mixed-halide alloying with the next halide
                t = (rA + rX) / (np.sqrt(2) * (rB + rX))
                mu = rB / rX
                gap = GAP0[(B, X)] + 0.35 * frac - 0.05 * (rA - 2.0)
                rows.append({"composition": f"{A}{B}({X}){'' if frac == 0 else f'+{frac:.2f} alloy'}",
                             "A": A, "B": B, "X": X, "t": t, "mu": mu, "gap": gap,
                             "efficiency": 100 * sq_efficiency(gap), "stable_geometry": 0.8 <= t <= 1.0 and 0.44 <= mu <= 0.9})
pv = pd.DataFrame(rows)
fig, ax = plt.subplots(figsize=(7.5, 5))
ok = pv[pv.stable_geometry]
ax.scatter(pv[~pv.stable_geometry].gap, pv[~pv.stable_geometry].efficiency, s=18, color="lightgrey", label="geometry unlikely")
sc = ax.scatter(ok.gap, ok.efficiency, c=ok.t, cmap="viridis", s=55, edgecolor="k", label="plausible perovskite")
best = ok.sort_values("efficiency", ascending=False).head(5)
for r in best.itertuples():
    ax.annotate(f"{r.A}{r.B}{r.X}₃", (r.gap, r.efficiency), textcoords="offset points", xytext=(6, 5), fontsize=9)
ax.set(xlabel="estimated band gap / eV", ylabel="limiting efficiency / %", title="Screening halide perovskites")
plt.colorbar(sc, label="tolerance factor t"); ax.legend(loc="lower left"); plt.show()
best[["composition", "t", "gap", "efficiency"]].round(2)
```

## MOFs: predicting gas uptake from pore descriptors

Metal–organic frameworks are defined by their geometry: surface area, pore size, void fraction. Those few numbers predict adsorption well, which is why geometric descriptors dominate MOF informatics {cite:p}`moosavi2020`.

```{code-cell} ipython3
n = 500
rng = np.random.default_rng(1)
asa = rng.uniform(500, 7000, n)                                  # accessible surface area, m²/g
pore = rng.uniform(3.5, 25, n)                                   # largest cavity diameter, Å
void = np.clip(0.2 + asa / 12000 + rng.normal(0, 0.06, n), 0.1, 0.95)
density = np.clip(2.2 - 1.8 * void + rng.normal(0, 0.05, n), 0.2, 2.0)
metal_strength = rng.choice([0.6, 1.0, 1.4], n)                   # open metal sites / functionalisation

# Low-pressure uptake needs strong binding sites and small pores; high pressure needs volume
uptake_1bar = 2 + 0.0012 * asa * metal_strength * np.exp(-(pore - 6) ** 2 / 60) + rng.normal(0, 0.6, n)
uptake_65bar = 2 + 0.0085 * asa * void + 2.5 * metal_strength + rng.normal(0, 1.5, n)
mof = pd.DataFrame({"ASA": asa, "pore_diameter": pore, "void_fraction": void, "density": density,
                    "site_strength": metal_strength, "uptake_1bar": uptake_1bar, "uptake_65bar": uptake_65bar})

from sklearn.ensemble import RandomForestRegressor
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
sc = axes[0].scatter(mof.ASA, mof.uptake_65bar, c=mof.void_fraction, cmap="cividis", s=16)
plt.colorbar(sc, ax=axes[0], label="void fraction"); axes[0].set(xlabel="surface area / m² g$^{-1}$", ylabel="uptake at 65 bar",
                                                                 title="High pressure: volume wins")
sc = axes[1].scatter(mof.pore_diameter, mof.uptake_1bar, c=mof.site_strength, cmap="Set1", s=16)
axes[1].set(xlabel="pore diameter / Å", ylabel="uptake at 1 bar", title="Low pressure: binding sites win")
for col, c in [("uptake_1bar", "#e5487a"), ("uptake_65bar", "#0f9d8a")]:
    rf = RandomForestRegressor(300, n_jobs=-1, random_state=0)
    X = mof[["ASA", "pore_diameter", "void_fraction", "density", "site_strength"]]
    p = cross_val_predict(rf, X, mof[col], cv=5)
    rf.fit(X, mof[col])
    axes[2].barh([f"{f}\n({col[-5:]})" for f in X.columns], rf.feature_importances_, color=c, alpha=0.7)
    print(f"{col}: CV R² = {np.corrcoef(p, mof[col])[0, 1] ** 2:.3f}")
axes[2].set_title("Different descriptors matter at different pressures")
plt.tight_layout(); plt.show()
```

### The deliverable-capacity trade-off

What matters in a tank is the **deliverable** capacity, the difference between storage and release pressures, and that peaks at intermediate binding strength: Sabatier again (Chapter 38).

```{code-cell} ipython3
mof["deliverable"] = mof.uptake_65bar - mof.uptake_1bar
fig, ax = plt.subplots(figsize=(7, 4.5))
for s, c, lab in [(0.6, "#3b82f6", "weak sites"), (1.0, "#f2a93b", "medium"), (1.4, "#e5487a", "strong sites")]:
    sub = mof[mof.site_strength == s]
    ax.scatter(sub.ASA, sub.deliverable, s=16, color=c, label=lab, alpha=0.7)
ax.set(xlabel="surface area / m² g$^{-1}$", ylabel="deliverable capacity", title="Strong binding stores gas but will not release it")
ax.legend(); plt.show()
```

## Exercises

**★ 1.** Compute the theoretical capacity of a silicon anode (Li₄.₄Si) and compare it with graphite (LiC₆).

**★★ 2.** Train the lifetime model using only the first 50 cycles. How much worse is it, and which features survive?

**★★★ 3.** Add a cost term (element prices) to the perovskite screen and find the best efficiency per unit cost.
