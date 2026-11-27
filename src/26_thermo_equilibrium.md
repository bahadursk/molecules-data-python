---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Thermodynamics and Equilibrium

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 12, 13 and 16.
```

Will a reaction go? How far? Which species dominate at a given pH? Thermodynamics answers with ΔG, K and mass balances, all of which are a few lines of Python.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
R = 8.314462618
```

## ΔH, ΔS and ΔG from tables

Standard data at 298 K (ΔfH° in kJ/mol, S° in J/(mol K)):

```{code-cell} ipython3
data = pd.DataFrame({
    "dHf": {"N2": 0, "H2": 0, "NH3": -45.9, "CH4": -74.6, "O2": 0, "CO2": -393.5, "H2O(l)": -285.8,
            "H2O(g)": -241.8, "CaCO3": -1207.6, "CaO": -634.9},
    "S":   {"N2": 191.6, "H2": 130.7, "NH3": 192.8, "CH4": 186.3, "O2": 205.2, "CO2": 213.8, "H2O(l)": 70.0,
            "H2O(g)": 188.8, "CaCO3": 91.7, "CaO": 38.1},
})

def reaction_thermo(reactants, products, T=298.15):
    dH = sum(n * data.dHf[s] for s, n in products.items()) - sum(n * data.dHf[s] for s, n in reactants.items())
    dS = sum(n * data.S[s] for s, n in products.items()) - sum(n * data.S[s] for s, n in reactants.items())
    dG = dH - T * dS / 1000
    return dH, dS, dG

for name, r, p in [("Haber", {"N2": 1, "H2": 3}, {"NH3": 2}),
                   ("methane combustion", {"CH4": 1, "O2": 2}, {"CO2": 1, "H2O(l)": 2}),
                   ("limestone calcination", {"CaCO3": 1}, {"CaO": 1, "CO2": 1})]:
    dH, dS, dG = reaction_thermo(r, p)
    print(f"{name:22s} ΔH = {dH:8.1f} kJ  ΔS = {dS:7.1f} J/K  ΔG = {dG:8.1f} kJ  K = {np.exp(-dG*1000/(R*298.15)):.2e}")
```

## Temperature dependence: where does ΔG change sign?

```{code-cell} ipython3
T = np.linspace(300, 1500, 300)
fig, ax = plt.subplots(figsize=(7, 4))
for name, r, p, c in [("Haber", {"N2": 1, "H2": 3}, {"NH3": 2}, "#6c4ad6"),
                      ("CaCO3 → CaO + CO2", {"CaCO3": 1}, {"CaO": 1, "CO2": 1}, "#e5487a")]:
    dH, dS, _ = reaction_thermo(r, p)
    dG = dH - T * dS / 1000
    ax.plot(T, dG, color=c, lw=2.5, label=name)
    ax.annotate(f"T* = {dH * 1000 / dS:.0f} K", (dH * 1000 / dS, 0), xytext=(0, 18), textcoords="offset points",
                ha="center", color=c, arrowprops=dict(arrowstyle="->", color=c))
ax.axhline(0, color="k", lw=0.8)
ax.fill_between(T, -200, 0, color="#0f9d8a", alpha=0.08)
ax.text(1300, -60, "spontaneous", color="#0f9d8a")
ax.set(xlabel="T / K", ylabel="ΔG° / kJ mol$^{-1}$", ylim=(-150, 150), title="Ellingham-style ΔG°(T)"); ax.legend()
plt.show()
```

## Equilibrium composition: the Haber process

For N₂ + 3H₂ ⇌ 2NH₃ starting from a 1:3 feed, with extent ξ, solve $K_p = \frac{x_{NH_3}^2}{x_{N_2}x_{H_2}^3}\left(\frac{P^\circ}{P}\right)^2$.

```{code-cell} ipython3
dH, dS, _ = reaction_thermo({"N2": 1, "H2": 3}, {"NH3": 2})

def nh3_fraction(T, P_bar):
    K = np.exp(-(dH * 1000 - T * dS) / (R * T))
    def f(xi):
        n = np.array([1 - xi, 3 - 3 * xi, 2 * xi]); x = n / n.sum()
        return np.log(x[2]**2 / (x[0] * x[1]**3) / P_bar**2) - np.log(K)
    xi = brentq(f, 1e-9, 1 - 1e-9)
    return 2 * xi / (4 - 2 * xi)

Ts = np.linspace(500, 900, 60)
fig, ax = plt.subplots(figsize=(7, 4))
for P, c in zip([1, 50, 100, 200, 400], plt.cm.viridis(np.linspace(0.1, 0.9, 5))):
    ax.plot(Ts - 273.15, [100 * nh3_fraction(t, P) for t in Ts], color=c, lw=2.5, label=f"{P} bar")
ax.axvspan(400, 500, color="grey", alpha=0.1); ax.text(405, 85, "industrial\nwindow", fontsize=9)
ax.set(xlabel="T / °C", ylabel="NH₃ at equilibrium / mol %", title="Le Chatelier in numbers"); ax.legend()
plt.show()
```

## Acid–base speciation diagrams

For a polyprotic acid H₃A, the fraction of each species depends only on pH and the pKa values.

```{code-cell} ipython3
def fractions(pH, pKas):
    H = 10.0 ** -np.asarray(pH)
    Ka = 10.0 ** -np.asarray(pKas)
    n = len(Ka)
    terms = [H ** (n - i) * np.prod(Ka[:i]) for i in range(n + 1)]
    total = sum(terms)
    return [t / total for t in terms]

pH = np.linspace(0, 14, 500)
systems = {"phosphoric acid": [2.15, 7.20, 12.35], "citric acid": [3.13, 4.76, 6.40], "glycine": [2.34, 9.60]}
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharey=True)
for ax, (name, pk) in zip(axes, systems.items()):
    for i, (f, c) in enumerate(zip(fractions(pH, pk), ["#e5487a", "#f2a93b", "#0f9d8a", "#6c4ad6"])):
        ax.plot(pH, f, color=c, lw=2.5, label=f"H{len(pk)-i}A" if len(pk) - i else "A")
        ax.fill_between(pH, f, alpha=0.12, color=c)
    for p in pk:
        ax.axvline(p, ls=":", color="grey")
    ax.set(title=name, xlabel="pH"); ax.legend(fontsize=8)
axes[0].set_ylabel("fraction"); plt.tight_layout(); plt.show()
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
At physiological pH 7.4, phosphate is a roughly 60:40 mixture of HPO₄²⁻ and H₂PO₄⁻, which is why it is such a good biological buffer.
```

## Exact titration curves by charge balance

Instead of piecewise approximations, solve the full charge balance at every point:

$$[H^+] + [Na^+] = [OH^-] + \sum_i i\,[H_{n-i}A^{i-}]$$

```{code-cell} ipython3
def titration(pKas, Ca=0.1, Va=25.0, Cb=0.1, Vb=np.linspace(0, 80, 400)):
    out = []
    for v in Vb:
        V = Va + v
        cA, cNa = Ca * Va / V, Cb * v / V
        def charge(pH):
            H = 10 ** -pH
            f = fractions(pH, pKas)
            anion = sum(i * fi for i, fi in enumerate(f)) * cA
            return H + cNa - 1e-14 / H - anion
        out.append(brentq(charge, -1, 15))
    return Vb, np.array(out)

fig, ax = plt.subplots(figsize=(7, 4))
for name, pk, c in [("HCl (strong)", [-7], "#e5487a"), ("acetic", [4.76], "#0f9d8a"), ("phosphoric", [2.15, 7.20, 12.35], "#6c4ad6")]:
    Vb, ph = titration(pk)
    ax.plot(Vb, ph, color=c, lw=2.5, label=name)
ax.set(xlabel="V(NaOH 0.1 M) / mL", ylabel="pH", title="Exact titration curves of 25 mL 0.1 M acids"); ax.legend()
plt.show()
```

## A simple phase diagram: water

```{code-cell} ipython3
T_tp, P_tp = 273.16, 611.657                       # triple point (K, Pa)
Hvap, Hsub, Hfus = 45.05e3, 51.06e3, 6.01e3        # J/mol near the triple point
Tl = np.linspace(273.16, 647, 300)
Ts = np.linspace(200, 273.16, 100)
P_lv = P_tp * np.exp(-Hvap / R * (1 / Tl - 1 / T_tp))
P_sv = P_tp * np.exp(-Hsub / R * (1 / Ts - 1 / T_tp))
P_sl = np.linspace(P_tp, 1e9, 100)
T_sl = T_tp + (-1.63e-6 * 273.16 / Hfus) * (P_sl - P_tp) * 1  # Clapeyron, ΔV_fus < 0 for water

fig, ax = plt.subplots(figsize=(6.5, 4.8))
ax.semilogy(Tl, P_lv, color="#3b82f6", lw=2.5)
ax.semilogy(Ts, P_sv, color="#6c4ad6", lw=2.5)
ax.semilogy(T_sl, P_sl, color="#0f9d8a", lw=2.5)
ax.plot(T_tp, P_tp, "ko"); ax.annotate("triple point", (T_tp, P_tp), xytext=(10, -15), textcoords="offset points")
ax.plot(647, 2.2e7, "r*", ms=14); ax.annotate("critical point", (647, 2.2e7), xytext=(-90, 10), textcoords="offset points")
for txt, xy in [("ICE", (225, 1e5)), ("LIQUID", (400, 1e7)), ("VAPOUR", (450, 1e2))]:
    ax.text(*xy, txt, fontsize=13, weight="bold", alpha=0.6)
ax.set(xlabel="T / K", ylabel="P / Pa", xlim=(200, 700), ylim=(1, 1e9), title="Phase diagram of water (Clausius–Clapeyron)")
plt.show()
```

## Exercises

**★ 1.** Calculate ΔG° and K at 298 K for the combustion of methane to gaseous water.

**★★ 2.** Plot the speciation diagram of carbonic acid (pKa 6.35, 10.33) and find the pH at which [HCO₃⁻] is maximal.

**★★★ 3.** Find the equilibrium composition of the water-gas shift reaction CO + H₂O ⇌ CO₂ + H₂ at 800 K, starting from equal moles of CO and H₂O (look up the data you need).
