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

A good figure is the fastest way to communicate a result. A reader takes in the shape of a titration curve or a periodic trend in a second, while the same numbers in a table take minutes to digest and may still hide the pattern. **Matplotlib** is Python's foundational plotting library: it turns arrays of numbers into lines, points, bars and images, and gives you control over every detail, from axis labels to the exact size of the figure for a journal column. Seaborn, pandas and many chemistry packages draw with Matplotlib underneath, so what you learn here applies everywhere. This chapter starts from the most basic line plot and builds up to multi-panel figures, colour maps, energy diagrams and publication-ready output.

By convention, the plotting part of the library is imported as `plt`, and NumPy (Chapter 12) is used to make the data:

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
```

## Your first plot

The basic idea of a line plot is simple: you give Matplotlib a list of x values and a list of y values of the same length, and it draws points at each (x, y) pair joined by straight lines. `plt.plot(x, y)` draws the line, `plt.xlabel` and `plt.ylabel` name the axes, and `plt.show()` displays the figure. Here are the boiling points of the first five straight-chain alkanes against the number of carbon atoms:

```{code-cell} ipython3
n_carbon = [1, 2, 3, 4, 5]
bp = [-161.5, -88.6, -42.1, -0.5, 36.1]     # °C, methane to pentane
plt.plot(n_carbon, bp, marker="o")
plt.xlabel("Number of carbon atoms")
plt.ylabel("Boiling point / °C")
plt.show()
```

`marker="o"` draws a dot at each data point as well as the line, which is good practice for measured data so the reader can see where the real values are. The rising trend reflects the growing London dispersion forces between longer chains.

## Two ways to plot

Matplotlib offers two styles of writing the same figure. The **pyplot** style, used above, calls functions such as `plt.plot` and `plt.xlabel` that act on "the current figure"; it is quick for a single plot. The **object-oriented** style starts with `fig, ax = plt.subplots()`, which creates a `Figure` (the whole canvas) and an `Axes` (one plotting area with its own axes), and then calls methods on `ax`. Because you hold a name for each axes, you can control several panels independently, and the code stays clear as figures grow. Use pyplot for quick looks and the object-oriented style for anything you will keep. Here the vapour pressure of water from the Magnus approximation is plotted both ways.

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

In the object-oriented version, `ax.set(...)` sets several properties at once, `ax.axhline` draws a horizontal reference line at atmospheric pressure, `ax.annotate` adds a label with an arrow, and `label=` together with `ax.legend()` produces a legend. Lines are customised with keyword arguments such as `color`, `lw` (line width) and `ls` (line style, `"--"` for dashed).

## Scatter plots with colour and size

A **scatter plot** draws each point as a separate marker without joining lines. It is the right choice when the points are separate observations rather than samples of a continuous curve, or when you want to encode extra information in each marker's colour or size. `ax.scatter(x, y, s=..., color=...)` sets the marker size `s` (in points squared) and colour. A minimal example shows density against molar mass for some liquids:

```{code-cell} ipython3
names = ["water", "ethanol", "hexane", "toluene", "chloroform"]
molar_mass = [18.0, 46.1, 86.2, 92.1, 119.4]
density = [0.997, 0.789, 0.655, 0.867, 1.489]
fig, ax = plt.subplots()
ax.scatter(molar_mass, density, s=80, color="#6c4ad6")
for x, y, n in zip(molar_mass, density, names):
    ax.text(x + 2, y, n, fontsize=9)
ax.set(xlabel="Molar mass / g mol$^{-1}$", ylabel="Density / g mL$^{-1}$")
plt.show()
```

There is no simple trend: density depends on how tightly molecules pack and on heavy atoms such as chlorine, not on molar mass alone. Text placed with `ax.text` labels each point; `$^{-1}$` uses Matplotlib's built-in maths notation for superscripts.

Colour becomes a powerful extra dimension when it shows a category. The periodic trend in first ionisation energy, coloured by block, makes the shell structure of the atom visible. A Boolean mask (`block == b`, Chapter 12) selects the elements of each block in turn:

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

Many results are best shown as several panels side by side that share an axis, for example a titration curve and its derivative. `plt.subplots(rows, cols)` creates a grid of axes and returns them as an array (or, as here, unpacked into separate names). `sharex=True` makes the panels use the same x range, so features line up, and `plt.tight_layout()` adjusts the spacing so labels do not overlap. A two-panel figure needs very little code:

```{code-cell} ipython3
t = np.linspace(0, 10, 50)
fig, (left, right) = plt.subplots(1, 2, figsize=(8, 3))
left.plot(t, np.exp(-0.3 * t)); left.set_title("[A] decays")
right.plot(t, 1 - np.exp(-0.3 * t), color="#e5487a"); right.set_title("[B] grows")
plt.tight_layout(); plt.show()
```

The titration of acetic acid with sodium hydroxide makes a richer example. The pH is computed piecewise: the weak-acid formula at the start, the Henderson–Hasselbalch equation in the buffer region, the hydrolysis of acetate at equivalence and excess hydroxide after it. `np.gradient` then gives the first derivative, whose sharp peak locates the equivalence point more precisely than the curve itself. A shaded band (`axhspan`) shows where the phenolphthalein indicator changes colour.

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

A **colormap** translates numbers into colours, so that a 2D grid of values, such as an electron density or a heat map, can be shown as an image. `ax.imshow(grid, cmap=...)` draws the grid, and `fig.colorbar` adds a key. The choice of colormap is not cosmetic. A good map changes brightness evenly with the value, so equal steps in the data look like equal steps in colour, and it remains readable in greyscale and to colour-blind readers.

A first image needs only a 2D array:

```{code-cell} ipython3
grid = np.add.outer(np.arange(5), np.arange(5))     # a 5 × 5 grid of values 0..8
fig, ax = plt.subplots(figsize=(3, 3))
im = ax.imshow(grid, cmap="viridis")
fig.colorbar(im, ax=ax)
plt.show()
```

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

Here the slice of a 2pₓ orbital is shown four ways. The probability density $|\psi|^2$ is always positive, so a sequential map suits it. The wavefunction $\psi$ itself changes sign across the nodal plane, so a diverging map with white at zero shows the two lobes of opposite phase clearly.

## Reaction-coordinate diagrams

Matplotlib is not limited to plotting data; it can draw schematic diagrams from simple pieces. A reaction-coordinate (energy-profile) diagram consists of short horizontal bars for the energies of stable species, joined by curves over the barriers. The helper function below draws a bar for each energy level and connects neighbouring levels with a smooth S-shaped curve, $3t^2 - 2t^3$, which starts and ends flat. Calling it twice compares an uncatalysed path with a catalysed one that passes through an intermediate. Hiding the top and right axis lines ("spines") gives a cleaner, textbook look.

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

Spectroscopists often want two scales on one plot, wavelength and photon energy, because absorption is measured in nanometres but interpreted in electronvolts. `ax.secondary_xaxis("top", functions=(forward, inverse))` adds a second axis that is computed from the first through a pair of functions, here $E = 1240/\lambda$ with $E$ in eV and $\lambda$ in nm. An **inset** is a small axes placed inside a larger one, created with `ax.inset_axes([x, y, width, height])` in fractions of the parent, and is ideal for magnifying a weak band. `axvspan` shades the visible region.

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

Journals ask for figures of a fixed width (a single column is about 8.5 cm, or 3.3 inches), with readable fonts and high resolution. `figsize` is given in inches, `dpi` sets dots per inch on screen, and `fig.savefig` writes the figure to a file whose format is chosen from the extension. A **style** changes many defaults at once (grid, fonts, colours); `plt.style.context(name)` applies one only inside a `with` block, so the rest of the notebook is unaffected. `plt.style.available` lists the styles installed:

```{code-cell} ipython3
print(plt.style.available[:8])
```

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

```{admonition} Summary
:class: tip
- `plt.plot(x, y)` draws lines, `scatter` draws separate points; always label axes as *quantity / unit*.
- Prefer the object-oriented style: `fig, ax = plt.subplots()`, then `ax.plot`, `ax.set(...)`, `ax.legend()`.
- `plt.subplots(rows, cols, sharex=True)` builds multi-panel figures; `tight_layout` fixes spacing.
- Use colour for an extra dimension; choose perceptually uniform colormaps (sequential for magnitudes, diverging for signed values).
- Annotations, shaded spans, secondary axes and insets add context.
- Save with `fig.savefig`: PDF/SVG for line art, PNG/TIFF at 300–600 dpi for images.
```

## Exercises

**★ 1.** Plot the Maxwell–Boltzmann speed distribution of N₂ at 200, 300 and 600 K on one axis with a legend.

**★★ 2.** Make a 2 × 2 subplot of the 1s, 2s, 3s and 4s radial probability distributions of hydrogen.

**★★★ 3.** Draw a molecular-orbital diagram of O₂ showing its two unpaired π* electrons.
