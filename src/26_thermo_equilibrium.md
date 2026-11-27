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

Will a reaction go? How far? Which species dominate at a given pH? Kinetics (Chapter 25) tells you how fast a reaction approaches its destination; **thermodynamics** tells you where that destination is. The central quantity is the **Gibbs energy change**, $\Delta G = \Delta H - T\Delta S$, which combines the heat released or absorbed ($\Delta H$, the enthalpy change) with the change in disorder ($\Delta S$, the entropy change). A negative $\Delta G^\circ$ means products are favoured at equilibrium, and its size sets the **equilibrium constant** through $\Delta G^\circ = -RT\ln K$. From $K$, together with *mass balances* (atoms are conserved) and *charge balances* (solutions are neutral), you can compute the full equilibrium composition of a gas mixture or a buffer. All of this is a few lines of Python. This chapter computes reaction thermodynamics from tables, follows how they change with temperature, solves gas-phase and acid–base equilibria, and draws a phase diagram.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import brentq, fsolve
R = 8.314462618
```

## ΔH, ΔS and ΔG from tables

Thermodynamic tables list, for each substance, its **standard enthalpy of formation** ΔfH° (the enthalpy change when one mole is formed from its elements in their standard states, zero for the elements themselves) and its **standard molar entropy** S°. Because enthalpy and entropy are state functions, the values for any reaction follow by subtraction, products minus reactants, each multiplied by its stoichiometric coefficient:

$$\Delta H^\circ = \sum_\text{products} n\,\Delta_f H^\circ - \sum_\text{reactants} n\,\Delta_f H^\circ$$

and likewise for $\Delta S^\circ$. A minimal example by hand, the formation of liquid water from H₂ and ½O₂:

```{code-cell} ipython3
dH_w = -285.8 - (0 + 0.5 * 0)                          # kJ/mol
dS_w = 70.0 - (130.7 + 0.5 * 205.2)                    # J/(mol K)
dG_w = dH_w - 298.15 * dS_w / 1000                     # note the J → kJ conversion
print(f"ΔH = {dH_w} kJ, ΔS = {dS_w:.1f} J/K, ΔG = {dG_w:.1f} kJ")
```

The result, −237 kJ mol⁻¹, is the tabulated Gibbs energy of formation of water. Storing the table in a pandas DataFrame and writing one function turns this into a calculator for any reaction. Standard data at 298 K (ΔfH° in kJ/mol, S° in J/(mol K)):

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

The Haber process and methane combustion are favourable at 298 K. Limestone calcination has a large positive ΔG°, so CaCO₃ is stable at room temperature, but its ΔS° is large and positive because a gas is released, which suggests heating will change the answer.

## Temperature dependence: where does ΔG change sign?

If $\Delta H$ and $\Delta S$ change little with temperature (a reasonable first approximation), $\Delta G^\circ(T) = \Delta H^\circ - T\Delta S^\circ$ is a straight line in $T$ with slope $-\Delta S^\circ$. It crosses zero at the **crossover temperature** $T^* = \Delta H^\circ/\Delta S^\circ$. Reactions that release heat but lose entropy, such as the Haber synthesis (4 gas molecules become 2), are favoured below $T^*$; reactions that absorb heat but gain entropy, such as calcination, are favoured above it. Plots of ΔG° against T like this one are called Ellingham diagrams in metallurgy.

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

$K$ tells you the ratio of products to reactants at equilibrium, but a chemical engineer wants the actual composition. The **extent of reaction** ξ counts how many times the reaction has occurred: starting from 1 mol N₂ and 3 mol H₂, after extent ξ there are $1-\xi$, $3-3\xi$ and $2\xi$ moles, so a single unknown describes the whole mixture. For gases, $K_p$ is written in mole fractions $x$ and total pressure $P$, and because the reaction reduces the number of gas molecules, pressure enters the expression. The equilibrium ξ is the root of "expression minus K", found with `brentq` (Chapter 16). Taking logarithms of both sides keeps the numbers well scaled. For N₂ + 3H₂ ⇌ 2NH₃ starting from a 1:3 feed, with extent ξ, solve $K_p = \frac{x_{NH_3}^2}{x_{N_2}x_{H_2}^3}\left(\frac{P^\circ}{P}\right)^2$.

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

The curves show **Le Chatelier's principle** quantitatively: the yield of ammonia falls with temperature (the reaction is exothermic) and rises with pressure (fewer gas molecules on the product side). Industry runs at 400–500 °C despite the lower yield, because the reaction is far too slow at low temperature: a compromise between thermodynamics and kinetics.

## Acid–base speciation diagrams

A **polyprotic acid** such as phosphoric acid, H₃PO₄, can lose several protons in turn, each step with its own pKa. In solution all the forms (H₃A, H₂A⁻, HA²⁻, A³⁻) coexist, and their **fractions** (often written α) depend only on pH and the pKa values. Each fraction is one term of a sum divided by the whole sum: the term for the species that has lost $i$ protons is $[\mathrm{H^+}]^{n-i} K_1 K_2 \cdots K_i$. A monoprotic acid has just two terms, which gives the familiar rule that the acid is half dissociated when pH = pKa:

```{code-cell} ipython3
pKa, pH_demo = 4.76, np.array([3.76, 4.76, 5.76])
H_demo = 10.0 ** -pH_demo
alpha_A = 10**-pKa / (H_demo + 10**-pKa)          # fraction of acetate
print(alpha_A.round(3))                            # about 0.09, 0.50, 0.91
```

One pH unit either side of the pKa moves the balance to roughly 10:90 or 90:10. The general function handles any number of protons, and a speciation diagram plots every fraction against pH.

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

Chapters 5 and 13 built titration curves piecewise, switching formulas at the equivalence point. A more powerful approach uses one equation valid everywhere: the **charge balance**, which says that a solution is electrically neutral, so the total positive charge equals the total negative charge. Every concentration in it can be written in terms of $[\mathrm{H^+}]$ using the speciation fractions, so at each volume of base there is one unknown, the pH, and `brentq` finds it. The same code then handles strong, weak and polyprotic acids without any special cases. Instead of piecewise approximations, solve the full charge balance at every point:

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

A **phase diagram** shows which phase of a substance (solid, liquid or gas) is stable at each temperature and pressure. The lines are where two phases coexist in equilibrium, and their shape follows from thermodynamics. The **Clausius–Clapeyron equation**, $\ln(P/P_0) = -\frac{\Delta H}{R}\left(\frac{1}{T} - \frac{1}{T_0}\right)$, gives the liquid–vapour and solid–vapour lines from the enthalpies of vaporisation and sublimation, starting at the **triple point** where all three phases meet. The solid–liquid line follows from the Clapeyron equation, $dP/dT = \Delta H_\text{fus}/(T\Delta V_\text{fus})$; for water, ice is less dense than liquid, so $\Delta V_\text{fus}$ is negative and the melting line leans backwards, which is why pressure melts ice. A logarithmic pressure axis shows the whole range from 1 Pa to 1 GPa.

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

```{admonition} Summary
:class: tip
- Reaction ΔH° and ΔS° are products minus reactants from tables; $\Delta G^\circ = \Delta H^\circ - T\Delta S^\circ$ and $K = e^{-\Delta G^\circ/RT}$.
- ΔG° changes sign at $T^* = \Delta H^\circ/\Delta S^\circ$.
- An extent of reaction ξ reduces an equilibrium to one unknown, solved with `brentq`.
- Speciation fractions depend only on pH and the pKa values; pH = pKa gives a 50:50 mixture.
- A charge balance gives exact titration curves for any acid.
- Clausius–Clapeyron and Clapeyron equations draw phase boundaries.
```

## Exercises

**★ 1.** Calculate ΔG° and K at 298 K for the combustion of methane to gaseous water.

**★★ 2.** Plot the speciation diagram of carbonic acid (pKa 6.35, 10.33) and find the pH at which [HCO₃⁻] is maximal.

**★★★ 3.** Find the equilibrium composition of the water-gas shift reaction CO + H₂O ⇌ CO₂ + H₂ at 800 K, starting from equal moles of CO and H₂O (look up the data you need).
