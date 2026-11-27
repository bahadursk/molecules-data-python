---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# SciPy: Fitting, Integration and Optimisation

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 12–13.
```

**SciPy** is a toolbox of numerical methods built on NumPy: curve fitting, integration, root finding, optimisation, signal processing, interpolation and physical constants. These are the everyday tools of physical and analytical chemistry.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants, optimize, integrate, signal, interpolate, stats
```

## Physical constants

```{code-cell} ipython3
print(constants.N_A, constants.k, constants.h, constants.R, constants.e, constants.c)
print(constants.physical_constants["Faraday constant"])
print(constants.convert_temperature(25, "Celsius", "Kelvin"))
```

## Curve fitting

### Michaelis–Menten enzyme kinetics

$$v = \frac{V_\max [S]}{K_M + [S]}$$

```{code-cell} ipython3
S = np.array([0.5, 1, 2, 4, 8, 16, 32, 64])                     # mM
rng = np.random.default_rng(3)
v_obs = 12.0 * S / (5.0 + S) * (1 + rng.normal(0, 0.04, S.size))  # μM/s

def mm(S, Vmax, KM):
    return Vmax * S / (KM + S)

popt, pcov = optimize.curve_fit(mm, S, v_obs, p0=[10, 3])
perr = np.sqrt(np.diag(pcov))
print(f"Vmax = {popt[0]:.2f} ± {perr[0]:.2f} μM/s,  KM = {popt[1]:.2f} ± {perr[1]:.2f} mM")

Sf = np.linspace(0, 70, 300)
fig, ax = plt.subplots(figsize=(6, 4))
ax.plot(S, v_obs, "o", color="#e5487a", ms=8, label="data")
ax.plot(Sf, mm(Sf, *popt), color="#6c4ad6", lw=2, label="fit")
ax.axhline(popt[0], ls=":", color="grey"); ax.axvline(popt[1], ls=":", color="grey")
ax.set(xlabel="[S] / mM", ylabel="v / μM s$^{-1}$", title="Michaelis–Menten fit"); ax.legend()
plt.show()
```

### Overlapping peaks

Deconvolute two overlapping Gaussian bands in a UV–Vis spectrum.

```{code-cell} ipython3
x = np.linspace(350, 650, 400)
def gauss(x, A, mu, s):
    return A * np.exp(-0.5 * ((x - mu) / s) ** 2)
def two_gauss(x, A1, m1, s1, A2, m2, s2):
    return gauss(x, A1, m1, s1) + gauss(x, A2, m2, s2)

y = two_gauss(x, 0.8, 470, 30, 0.5, 540, 25) + rng.normal(0, 0.01, x.size)
p, _ = optimize.curve_fit(two_gauss, x, y, p0=[0.7, 460, 20, 0.4, 550, 20])
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(x, y, ".", color="grey", ms=3, label="data")
ax.fill_between(x, gauss(x, *p[:3]), color="#0f9d8a", alpha=0.4, label=f"band 1, λ={p[1]:.0f} nm")
ax.fill_between(x, gauss(x, *p[3:]), color="#f2a93b", alpha=0.4, label=f"band 2, λ={p[4]:.0f} nm")
ax.plot(x, two_gauss(x, *p), "k", lw=1.5, label="sum")
ax.set(xlabel="λ / nm", ylabel="Absorbance"); ax.legend(); plt.show()
```

## Numerical integration

### Peak areas

```{code-cell} ipython3
area1 = integrate.trapezoid(gauss(x, *p[:3]), x)
area2 = integrate.simpson(gauss(x, *p[3:]), x=x)
print(f"area ratio band1/band2 = {area1 / area2:.2f}")
```

### Integrating a function: heat capacity

The enthalpy change when heating N₂ from 298 to 1000 K, with $C_p(T) = a + bT + cT^{-2}$:

```{code-cell} ipython3
a, b, c = 28.58, 3.77e-3, -0.50e5        # J/(mol K)
dH, err = integrate.quad(lambda T: a + b * T + c / T**2, 298, 1000)
print(f"ΔH = {dH / 1000:.2f} kJ/mol (error estimate {err:.1e})")
```

## Root finding: equilibrium concentrations

0.10 mol of N₂O₄ in 1.0 L dissociates: N₂O₄ ⇌ 2 NO₂, Kc = 4.6 × 10⁻³. With x the amount dissociated, $K_c = (2x)^2 / (0.10 - x)$.

```{code-cell} ipython3
Kc, C0 = 4.6e-3, 0.10
f = lambda x: (2 * x) ** 2 / (C0 - x) - Kc
x = optimize.brentq(f, 1e-12, C0 - 1e-12)
print(f"x = {x:.4f} M, [N2O4] = {C0 - x:.4f} M, [NO2] = {2 * x:.4f} M, α = {x / C0:.1%}")
```

## Optimisation: finding minima

The Lennard-Jones potential between two argon atoms:

```{code-cell} ipython3
eps, sigma = 0.997, 3.40          # kJ/mol, Å
LJ = lambda r: 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
res = optimize.minimize_scalar(LJ, bounds=(3, 6), method="bounded")
print(f"r_min = {res.x:.3f} Å (theory 2^(1/6)σ = {2 ** (1/6) * sigma:.3f}), E_min = {res.fun:.3f} kJ/mol")
```

A multidimensional example, the geometry of a three-atom Lennard-Jones cluster:

```{code-cell} ipython3
def cluster_energy(flat):
    xyz = flat.reshape(-1, 3)
    E = 0
    for i in range(len(xyz)):
        for j in range(i + 1, len(xyz)):
            E += LJ(np.linalg.norm(xyz[i] - xyz[j]))
    return E

start = np.random.default_rng(0).normal(0, 2, 9) + np.array([0, 0, 0, 4, 0, 0, 2, 3, 0])
opt = optimize.minimize(cluster_energy, start, method="BFGS")
xyz = opt.x.reshape(-1, 3)
d = [np.linalg.norm(xyz[i] - xyz[j]) for i, j in [(0, 1), (1, 2), (0, 2)]]
print("E =", round(opt.fun, 3), "kJ/mol; distances:", np.round(d, 3), "-> equilateral triangle")
```

## Signal processing: smoothing and peak finding

```{code-cell} ipython3
t = np.linspace(0, 20, 2000)                       # chromatogram, minutes
chrom = sum(h * np.exp(-0.5 * ((t - c) / w) ** 2) for c, w, h in
            [(3.1, 0.12, 1.0), (5.4, 0.15, 0.6), (5.9, 0.15, 0.4), (9.8, 0.25, 0.8), (14.2, 0.3, 0.3)])
noisy = chrom + 0.02 * t / 20 + rng.normal(0, 0.015, t.size)
smooth = signal.savgol_filter(noisy, window_length=41, polyorder=3)
peaks, props = signal.find_peaks(smooth, height=0.1, prominence=0.05)

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.plot(t, noisy, color="lightgrey", label="raw")
ax.plot(t, smooth, color="#6c4ad6", label="Savitzky–Golay")
ax.plot(t[peaks], smooth[peaks], "v", color="#e5487a", ms=10, label="peaks")
for pk in peaks:
    ax.text(t[pk], smooth[pk] + 0.05, f"{t[pk]:.2f}", ha="center", fontsize=8)
ax.set(xlabel="Retention time / min", ylabel="Detector signal", title="HPLC chromatogram"); ax.legend()
plt.show()
```

## Interpolation

```{code-cell} ipython3
T_tab = np.array([0, 20, 40, 60, 80, 100])              # °C
rho_tab = np.array([0.99984, 0.99821, 0.99222, 0.98320, 0.97179, 0.95835])  # g/mL water
spl = interpolate.CubicSpline(T_tab, rho_tab)
print(f"density at 37 °C ≈ {spl(37):.5f} g/mL")
```

## Statistics: is the difference real?

```{code-cell} ipython3
method_A = np.array([10.12, 10.08, 10.15, 10.11, 10.09])   # % Fe
method_B = np.array([10.21, 10.18, 10.25, 10.19, 10.23])
tstat, pval = stats.ttest_ind(method_A, method_B)
print(f"t = {tstat:.2f}, p = {pval:.4f} ->", "significant" if pval < 0.05 else "not significant")
lin = stats.linregress([1, 2, 3, 4, 5], [0.101, 0.200, 0.302, 0.401, 0.499])
print(f"calibration slope {lin.slope:.4f}, R² = {lin.rvalue**2:.5f}")
```

## Exercises

**★ 1.** Use `optimize.brentq` to find the pH of 0.010 M acetic acid exactly (no approximation).

**★★ 2.** Fit the integrated second-order rate law $1/[A] = 1/[A]_0 + kt$ to simulated data and report k with its uncertainty.

**★★★ 3.** Find the global minimum geometry of a 7-atom Lennard-Jones cluster using `optimize.basinhopping`.
