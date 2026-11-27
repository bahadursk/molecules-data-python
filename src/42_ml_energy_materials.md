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

Energy technologies are limited by materials: how much charge a battery electrode can store and how long it lasts, how efficiently a solar absorber converts light, how much gas a porous framework can hold. Each has become a machine-learning problem, from battery lifetime prediction {cite:p}`severson2019` to the diversity of the MOF universe {cite:p}`moosavi2020`. These problems share a pattern. Some quantities follow directly from simple physics, such as the theoretical capacity of a battery electrode from its formula or the maximum efficiency of a solar cell from its band gap, and computing them first sets the limits within which any real material must fall. Other quantities, such as how many cycles a battery will survive or how much methane a porous solid adsorbs, depend on complicated processes that are expensive to measure, and here models trained on data predict them from cheap measurements or descriptors. This chapter works through batteries (capacity, energy density, lifetime prediction), photovoltaics (the Shockley–Queisser limit and a perovskite screen) and metal–organic frameworks (gas uptake and deliverable capacity), with simulated data throughout.

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

Here $n$ is the number of electrons transferred per formula unit, $F$ the Faraday constant (the charge of one mole of electrons, 96 485 C mol⁻¹) and $M$ the molar mass in g mol⁻¹. The factor 3.6 converts coulombs per gram into milliampere-hours per gram, since 1 mAh = 3.6 C. A light material that exchanges many electrons stores the most charge per gram. For LiCoO₂, with one lithium (and so one electron) per formula unit:

```{code-cell} ipython3
M_lco = 6.94 + 58.93 + 2 * 16.00
print(f"{1 * 96485.3 / (3.6 * M_lco):.0f} mAh/g")
```

pymatgen's `Composition(...).weight` gives the molar mass for any formula, so the calculation extends to a whole list of electrode materials.

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

Capacity alone is not the whole story. The energy stored is charge times voltage, so the specific energy in Wh kg⁻¹ is the capacity in mAh g⁻¹ multiplied by the cell voltage in volts (the factors of 1000 cancel). A high-voltage material with a modest capacity can store as much energy as a low-voltage one with a large capacity. A complete cell also contains an anode, electrolyte, separator and casing, which is why the code applies a rough factor of 0.4 to go from electrode to cell level. Plotting energy against cycle life on log axes, in the spirit of a **Ragone plot**, shows the central trade-off: the chemistries that store the most energy tend to last the fewest cycles. For a cell at 3.7 V with 180 mAh g⁻¹:

```{code-cell} ipython3
print(3.7 * 180, "Wh/kg at electrode level;", round(3.7 * 180 * 0.4), "Wh/kg as a rough cell-level estimate")
```

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

Severson and co-workers showed that the **change in discharge-capacity curves over the first 100 cycles** predicts total lifetime, long before any capacity fade is visible {cite:p}`severson2019`. Here we simulate cells with that structure and reproduce the workflow. Testing a battery to the end of its life, conventionally the cycle at which its capacity falls to 80 % of the initial value, can take months or years, so predicting lifetime from the first few weeks of data saves enormous time in development. Each simulated cell loses capacity by two hidden mechanisms: growth of the solid–electrolyte interphase (SEI), which consumes lithium and grows with the square root of cycle number, and loss of active material, which grows linearly. The rates differ from cell to cell, and the same SEI growth also raises the internal resistance, so the early data carry faint traces of the eventual lifetime. `np.argmax(cap < 0.8)` finds the first cycle at which the capacity drops below 80 %, because `argmax` of a Boolean array returns the position of the first `True`:

```{code-cell} ipython3
print(np.argmax(np.array([1.0, 0.9, 0.85, 0.79, 0.75]) < 0.8))
```

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

From the first 100 cycles of each cell, nine features are computed: capacities at cycles 10 and 100, their difference, the slope and curvature of a polynomial fit (Chapter 16), the variance of the cycle-to-cycle changes, the internal resistance and its change. Lifetime varies over a large range, so the model predicts $\log_{10}$ of the cycle life. The **elastic net** is a linear regression that combines the two penalties of Chapter 31, the ridge penalty, which shrinks coefficients, and the lasso penalty, which can set them exactly to zero, so it also selects features; `ElasticNetCV` chooses the strength of the penalties by cross-validation. Accuracy is reported as the **mean absolute percentage error** (MAPE), the average error as a percentage of the true lifetime.

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

The band gap of an absorber sets a hard ceiling on efficiency: too small and photons are wasted as heat, too large and most of the spectrum is not absorbed. A photon with energy below the band gap passes straight through the absorber. A photon with energy above it is absorbed, but the excited electron quickly relaxes to the bottom of the conduction band, so any energy beyond the gap is lost as heat. Every absorbed photon therefore delivers at most the gap energy, and the efficiency is a balance between the number of photons absorbed, which falls as the gap rises, and the energy each one delivers, which rises with the gap. The detailed-balance calculation of Shockley and Queisser gives a maximum of about 33 % near 1.3–1.4 eV for a single junction under sunlight.

The calculation below approximates sunlight as a black body at 5800 K, converts each wavelength to a photon energy with $E = hc/\lambda$, counts the photons above each gap by integrating with `np.trapezoid`, and applies simple empirical factors for the voltage and fill-factor losses, so its numbers are illustrative. A photon at 1100 nm has just enough energy for silicon:

```{code-cell} ipython3
print(f"{6.626e-34 * 2.998e8 / 1100e-9 / 1.602e-19:.2f} eV")
```

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

Halide perovskites have the formula ABX₃, with a large cation A (caesium or a small organic ion such as methylammonium, MA, or formamidinium, FA), a metal B (lead, tin or germanium) and a halide X. Whether a combination can form the perovskite structure is estimated from ionic radii by two geometric rules. **Goldschmidt's tolerance factor** $t = (r_A + r_X)/(\sqrt{2}(r_B + r_X))$ measures how well A fits into the cavity between the BX₆ octahedra, with values between about 0.8 and 1.0 favouring a perovskite; the **octahedral factor** $\mu = r_B/r_X$ checks that B fits inside its octahedron of halides. For CsPbI₃, using the radii in the table below (Å):

```{code-cell} ipython3
r_A, r_B, r_X = 1.88, 1.19, 2.20
print(f"t = {(r_A + r_X) / (np.sqrt(2) * (r_B + r_X)):.3f}, mu = {r_B / r_X:.3f}")
```

The band gaps come from a simple lookup for each metal–halide pair, shifted by halide alloying and the size of A; they mimic the known trends (the gap rises from iodide to bromide to chloride) but are not measured values.

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

Metal–organic frameworks are defined by their geometry: surface area, pore size, void fraction. Those few numbers predict adsorption well, which is why geometric descriptors dominate MOF informatics {cite:p}`moosavi2020`. A MOF is a crystalline network of metal ions or clusters linked by organic molecules, enclosing pores that can make up most of its volume; a single gram can have a surface area of several thousand square metres. Computer programs calculate the pore geometry of a hypothetical MOF in seconds, whereas a grand canonical Monte Carlo simulation of adsorption takes hours, so a model that maps geometry to uptake allows huge libraries to be screened. The simulated data below build in two physical regimes. At low pressure few molecules are adsorbed, and they sit at the strongest binding sites, so binding strength and pores close to the size of the molecule matter. At high pressure the pores fill, so the total pore volume, through surface area and void fraction, matters most.

```{code-cell} ipython3
n = 500
rng = np.random.default_rng(1)
asa = rng.uniform(500, 7000, n)                                  # accessible surface area, m²/g
pore = rng.uniform(3.5, 25, n)                                   # largest cavity diameter, Å
void = np.clip(0.2 + asa / 12000 + rng.normal(0, 0.06, n), 0.1, 0.95)
density = np.clip(2.2 - 1.8 * void + rng.normal(0, 0.05, n), 0.2, 2.0)
metal_strength = rng.choice([0.6, 1.0, 1.4], n)                   # open metal sites / functionalisation

# Low-pressure uptake needs strong binding sites and small pores; high pressure needs volume
# Strong sites fill early (low pressure) and saturate, so their benefit at high pressure levels off
uptake_1bar = 2 + 0.0032 * asa * metal_strength ** 2 * np.exp(-(pore - 6) ** 2 / 60) + rng.normal(0, 0.6, n)
uptake_65bar = 2 + 0.0085 * asa * void + 40 * (1 - np.exp(-3 * metal_strength)) + rng.normal(0, 1.5, n)
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

What matters in a tank is the **deliverable** capacity, the difference between storage and release pressures, and that suffers when binding is too strong: Sabatier again (Chapter 38).

```{code-cell} ipython3
mof["deliverable"] = mof.uptake_65bar - mof.uptake_1bar
fig, ax = plt.subplots(figsize=(7, 4.5))
for s, c, lab in [(0.6, "#3b82f6", "weak sites"), (1.0, "#f2a93b", "medium"), (1.4, "#e5487a", "strong sites")]:
    sub = mof[mof.site_strength == s]
    ax.scatter(sub.ASA, sub.deliverable, s=16, color=c, label=lab, alpha=0.7)
ax.set(xlabel="surface area / m² g$^{-1}$", ylabel="deliverable capacity", title="Strong binding stores gas but will not release it")
ax.legend(); plt.show()
mof.groupby("site_strength")[["uptake_1bar", "uptake_65bar", "deliverable"]].mean().round(1)
```

Strong sites give the highest storage at 65 bar, but they also hold on to so much gas at 1 bar that they deliver the least; weak and medium sites deliver about the same. More binding is useful only up to a point.

```{admonition} Summary
:class: tip
- Theoretical capacity follows from $nF/(3.6M)$; energy density multiplies capacity by voltage.
- Early-cycle features, especially changes in the capacity curve, predict battery lifetime; predict log(lifetime) with a regularised linear model.
- The Shockley–Queisser balance between absorbed photons and energy per photon sets an optimum band gap near 1.3–1.4 eV.
- Tolerance and octahedral factors screen perovskite compositions for plausible geometry.
- MOF uptake depends on binding sites at low pressure and pore volume at high pressure; deliverable capacity is the difference.
```

## Exercises

**★ 1.** Compute the theoretical capacity of a silicon anode (Li₄.₄Si) and compare it with graphite (LiC₆).

**★★ 2.** Train the lifetime model using only the first 50 cycles. How much worse is it, and which features survive?

**★★★ 3.** Add a cost term (element prices) to the perovskite screen and find the best efficiency per unit cost.
