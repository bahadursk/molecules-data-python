---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Matplotlib: Publication-Quality Figures

```{admonition} 🟢 Beginner · school & first-year
:class: level-1
Needs only Part I. NumPy (Chapter 12) helps but is not required.
```

A good figure is the fastest way to communicate a result. **Matplotlib** is Python's foundational plotting library; Seaborn, pandas and many chemistry packages draw with it underneath.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## Two ways to plot

```{code-cell} ipython3
T = np.linspace(250, 400, 100)
P_water = 0.61094 * np.exp(17.625 * (T - 273.15) / (T - 273.15 + 243.04))   # kPa, Magnus formula

# 1. pyplot style: quick
plt.plot(T, P_water)
plt.xlabel("T / K"); plt.ylabel("Vapour pressure / kPa")
plt.show()
```

```{code-cell} ipython3
# 2. object-oriented style: full control (recommended)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(T, P_water, color="#0f9d8a", lw=2.5, label="water")
ax.axhline(101.325, ls="--", color="grey")
ax.annotate("normal boiling point", xy=(373.1, 101.3), xytext=(300, 80),
            arrowprops=dict(arrowstyle="->"), fontsize=10)
ax.set(xlabel="T / K", ylabel="Vapour pressure / kPa", title="Vapour pressure of water")
ax.legend()
plt.show()
```

## Scatter plots with colour and size

The periodic trend in first ionisation energy, coloured by block.

```{code-cell} ipython3
Z = np.arange(1, 37)
IE = np.array([13.60, 24.59, 5.39, 9.32, 8.30, 11.26, 14.53, 13.62, 17.42, 21.56, 5.14, 7.65,
               5.99, 8.15, 10.49, 10.36, 12.97, 15.76, 4.34, 6.11, 6.56, 6.83, 6.75, 6.77, 7.43,
               7.90, 7.88, 7.64, 7.73, 9.39, 6.00, 7.90, 9.79, 9.75, 11.81, 14.00])
block = np.array(list("ssss" + "p" * 6 + "ss" + "p" * 6 + "ss" + "d" * 10 + "p" * 6))
colours = {"s": "#e5487a", "p": "#f2a93b", "d": "#0f9d8a"}
symbols = "H He Li Be B C N O F Ne Na Mg Al Si P S Cl Ar K Ca Sc Ti V Cr Mn Fe Co Ni Cu Zn Ga Ge As Se Br Kr".split()

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(Z, IE, color="lightgrey", zorder=0)
for b, c in colours.items():
    m = block == b
    ax.scatter(Z[m], IE[m], s=60, color=c, edgecolor="k", label=f"{b}-block")
for z, e, s in zip(Z, IE, symbols):
    if s in ("He", "Ne", "Ar", "Kr", "Li", "Na", "K"):
        ax.text(z, e + 0.7, s, ha="center", fontsize=9)
ax.set(xlabel="Atomic number", ylabel="First ionisation energy / eV")
ax.legend(); ax.set_title("Periodicity of ionisation energy")
plt.show()
```

## Subplots: a titration dashboard

```{code-cell} ipython3
Vb = np.linspace(0, 50, 500)
Ka, Ca, Va, Cb = 1.8e-5, 0.1, 25, 0.1

def weak_acid_pH(Vb):
    na, nb = Ca * Va, Cb * Vb
    V = Va + Vb
    if nb < 1e-12:
        return -np.log10(np.sqrt(Ka * Ca))
    if nb < na:
        return -np.log10(Ka) + np.log10(nb / (na - nb))
    if np.isclose(nb, na):
        Kb = 1e-14 / Ka
        return 14 + np.log10(np.sqrt(Kb * na / V))
    return 14 + np.log10((nb - na) / V)

pH = np.array([weak_acid_pH(v) for v in Vb])
dpH = np.gradient(pH, Vb)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharex=True)
ax1.plot(Vb, pH, color="#6c4ad6", lw=2.5)
ax1.axhspan(8.2, 10, color="#ff66cc", alpha=0.2, label="phenolphthalein range")
ax1.axvline(12.5, ls=":", color="k"); ax1.text(13, 3.5, "half-equivalence\npH = pKa", fontsize=9)
ax1.set(xlabel="V(NaOH) / mL", ylabel="pH", title="Acetic acid + NaOH"); ax1.legend(loc="lower right")
ax2.plot(Vb, dpH, color="#e5487a"); ax2.set(xlabel="V(NaOH) / mL", ylabel="dpH/dV", title="First derivative")
ax2.set_ylim(0, 5)
plt.tight_layout(); plt.show()
```

## Colormaps and colour theory

Choose colormaps with care: **sequential** for magnitudes, **diverging** for signed quantities, **qualitative** for categories. Avoid `jet`; prefer perceptually uniform maps such as `viridis`, `magma` and `cividis` (colour-blind friendly).

```{code-cell} ipython3
fig, axes = plt.subplots(1, 4, figsize=(11, 2.8))
x, y = np.meshgrid(np.linspace(-2, 2, 200), np.linspace(-2, 2, 200))
r = np.hypot(x, y)
psi_2p = x * np.exp(-r)                # a 2p_x orbital slice
for ax, cmap in zip(axes, ["viridis", "magma", "RdBu_r", "cividis"]):
    data = psi_2p if cmap == "RdBu_r" else psi_2p ** 2
    im = ax.imshow(data, cmap=cmap, extent=[-2, 2, -2, 2])
    ax.set_title(cmap); ax.axis("off")
fig.suptitle("2pₓ orbital: |ψ|² in sequential maps, ψ (signed) in a diverging map")
plt.show()
```

## Reaction-coordinate diagrams

```{code-cell} ipython3
def profile(levels, ax, colour, label):
    xs, ys = [], []
    for i, E in enumerate(levels):
        x = np.linspace(2 * i, 2 * i + 1, 10)
        ax.plot(x, [E] * 10, color=colour, lw=3)
        if i < len(levels) - 1:
            t = np.linspace(0, 1, 50)
            ax.plot(2 * i + 1 + t, E + (levels[i + 1] - E) * (3 * t**2 - 2 * t**3), color=colour, lw=1.2, ls="--")
    ax.plot([], [], color=colour, lw=3, label=label)

fig, ax = plt.subplots(figsize=(8, 4.5))
profile([0, 85, -30], ax, "#e5487a", "uncatalysed")
profile([0, 50, 10, 45, -30], ax, "#0f9d8a", "catalysed")
ax.set_xticks([0.5, 4.5, 8.5], ["reactants", "intermediate", "products"])
ax.set_ylabel("G / kJ mol$^{-1}$"); ax.legend(); ax.set_title("A catalyst lowers the activation barrier")
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
plt.show()
```

## Spectra with insets and secondary axes

```{code-cell} ipython3
wl = np.linspace(200, 800, 600)
abs_ = 1.2 * np.exp(-((wl - 260) / 15) ** 2) + 0.4 * np.exp(-((wl - 450) / 40) ** 2)
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(wl, abs_, color="k")
ax.set(xlabel="λ / nm", ylabel="Absorbance")
sec = ax.secondary_xaxis("top", functions=(lambda l: 1240 / l, lambda e: 1240 / e))
sec.set_xlabel("Photon energy / eV")
ax.axvspan(380, 750, color="gold", alpha=0.08)
inset = ax.inset_axes([0.55, 0.45, 0.4, 0.45])
m = (wl > 380) & (wl < 520)
inset.plot(wl[m], abs_[m], color="#6c4ad6"); inset.set_title("visible band", fontsize=9)
plt.show()
```

## Styling for journals

```{code-cell} ipython3
with plt.style.context("seaborn-v0_8-whitegrid"):
    fig, ax = plt.subplots(figsize=(3.3, 2.5), dpi=150)   # single journal column ≈ 3.3 in
    for n, c in zip([1, 2, 3], ["#0f9d8a", "#6c4ad6", "#e5487a"]):
        t = np.linspace(0, 5, 100)
        ax.plot(t, np.exp(-n * t), color=c, label=f"k = {n} s$^{{-1}}$")
    ax.set(xlabel="t / s", ylabel="[A]/[A]$_0$")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig("first_order_decay.pdf")     # vector graphics for print
    fig.savefig("first_order_decay.png", dpi=600)
plt.show()
```

```{tip}
Save line art as PDF or SVG (vector) and photos as PNG or TIFF at 300–600 dpi. Label axes as *quantity / unit*, following IUPAC.
```

## Exercises

**★ 1.** Plot the Maxwell–Boltzmann speed distribution of N₂ at 200, 300 and 600 K on one axis with a legend.

**★★ 2.** Make a 2 × 2 subplot of the 1s, 2s, 3s and 4s radial probability distributions of hydrogen.

**★★★ 3.** Draw a molecular-orbital diagram of O₂ showing its two unpaired π* electrons.
