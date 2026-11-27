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

NumPy gives you arrays and arithmetic, but science needs more: fitting a model to measurements, finding the area under a peak, solving an equilibrium expression that has no neat formula, or locating the lowest-energy geometry of a molecule. **SciPy** ("Scientific Python") is a toolbox of tested numerical methods built on NumPy that does exactly these jobs. Each kind of task lives in its own submodule: `optimize` for fitting, root finding and minimisation, `integrate` for areas and integrals, `signal` for smoothing and peak finding, `interpolate` for values between tabulated points, `stats` for statistical tests, and `constants` for the physical constants. These are the everyday tools of physical and analytical chemistry, and using them means you rely on algorithms that have been checked by thousands of scientists rather than on your own first attempt. This chapter takes each submodule in turn with a chemistry problem it solves.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import constants, optimize, integrate, signal, interpolate, stats
```

## Physical constants

Typing constants by hand invites mistakes in the fifth digit. `scipy.constants` provides the CODATA recommended values of the fundamental constants in SI units, so the Avogadro constant, Boltzmann constant, Planck constant, gas constant, elementary charge and speed of light are all one name away. The dictionary `physical_constants` gives each constant's value, unit and uncertainty.

```{code-cell} ipython3
print(constants.R)                     # gas constant, J/(mol K)
print(constants.h * constants.c / 500e-9, "J per 500 nm photon")
```

```{code-cell} ipython3
print(constants.N_A, constants.k, constants.h, constants.R, constants.e, constants.c)
print(constants.physical_constants["Faraday constant"])
print(constants.convert_temperature(25, "Celsius", "Kelvin"))
```

## Curve fitting

**Curve fitting** means finding the parameters of a model equation that make it agree best with measured data. "Best" usually means *least squares*: the parameters that minimise the sum of the squared differences between the measured values and the model's predictions. Fitting is how you turn raw data into physical quantities, such as a rate constant, an equilibrium constant or a peak position, together with an estimate of how precisely they are known. `optimize.curve_fit(model, x, y, p0=...)` needs a Python function `model(x, a, b, ...)`, the data, and starting guesses `p0`. It returns the best parameters `popt` and their *covariance matrix* `pcov`, whose diagonal holds the variances, so `np.sqrt(np.diag(pcov))` gives each parameter's standard error.

The most basic case fits a straight line, a Beer–Lambert calibration $A = \varepsilon l c$ with a small offset:

```{code-cell} ipython3
c = np.array([0.0, 0.1, 0.2, 0.3, 0.4])          # mM
A = np.array([0.003, 0.152, 0.297, 0.455, 0.601])
line = lambda c, slope, offset: slope * c + offset
popt, pcov = optimize.curve_fit(line, c, A)
print("slope, offset =", popt.round(4), "±", np.sqrt(np.diag(pcov)).round(4))
```

### Michaelis–Menten enzyme kinetics

The real power of `curve_fit` is that the model can be any non-linear equation. Enzyme rates follow the Michaelis–Menten equation, in which the rate $v$ rises with substrate concentration $[S]$ and levels off at a maximum $V_\max$; $K_M$ is the concentration at which the rate is half its maximum.

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

The dotted lines mark the fitted $V_\max$ and $K_M$. Older textbooks linearise this equation (the Lineweaver–Burk plot) so that it can be fitted with a straight line, but that distorts the errors; fitting the non-linear equation directly is more accurate.

### Overlapping peaks

When two absorption bands overlap, the measured spectrum is their sum and neither peak can be read off directly. **Deconvolution** fits a model made of several peak shapes and recovers each band's position, width and height. Here the model is the sum of two Gaussians with six parameters. Good starting guesses matter for non-linear fits with many parameters, because the fitting algorithm walks downhill from `p0` and can settle in the wrong valley if it starts too far away.

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

**Integration** finds the area under a curve. In chemistry this gives the amount of substance from a chromatographic peak, the relative number of protons from an NMR multiplet, or an enthalpy from a heat capacity. When you have data points rather than a formula, the area is approximated from the points: the **trapezoidal rule** joins neighbouring points with straight lines and adds up the trapezoids, and **Simpson's rule** uses parabolas through groups of three points, which is more accurate for smooth curves. A basic check on a shape whose area we know, a triangle of base 2 and height 1:

```{code-cell} ipython3
xs = [0, 1, 2]
ys = [0, 1, 0]
print(integrate.trapezoid(ys, xs))
```

### Peak areas

The areas of the two fitted bands are proportional to the amounts of the two absorbing species (if their absorptivities are equal):

```{code-cell} ipython3
area1 = integrate.trapezoid(gauss(x, *p[:3]), x)
area2 = integrate.simpson(gauss(x, *p[3:]), x=x)
print(f"area ratio band1/band2 = {area1 / area2:.2f}")
```

### Integrating a function: heat capacity

When you know the formula of the function, `integrate.quad` integrates it to high accuracy and also returns an estimate of the numerical error. Thermodynamics needs this constantly: the enthalpy change on heating is $\Delta H = \int_{T_1}^{T_2} C_p(T)\,dT$. The enthalpy change when heating N₂ from 298 to 1000 K, with $C_p(T) = a + bT + cT^{-2}$:

```{code-cell} ipython3
a, b, c = 28.58, 3.77e-3, -0.50e5        # J/(mol K)
dH, err = integrate.quad(lambda T: a + b * T + c / T**2, 298, 1000)
print(f"ΔH = {dH / 1000:.2f} kJ/mol (error estimate {err:.1e})")
```

## Root finding: equilibrium concentrations

A **root** of a function is a value of $x$ at which $f(x) = 0$. Many chemistry problems become root-finding problems: an equilibrium expression rearranged as "expression minus K equals zero" is satisfied at its root. Simple cases have a formula, such as the quadratic of Chapter 6, but most do not. `optimize.brentq(f, a, b)` finds a root between two values `a` and `b` where `f` has opposite signs; it is guaranteed to converge, so you only need to bracket the answer by physical reasoning. A basic example finds $\sqrt{2}$ as the root of $x^2 - 2$:

```{code-cell} ipython3
print(optimize.brentq(lambda x: x**2 - 2, 0, 2))
```

For the equilibrium below, the amount dissociated $x$ must lie between 0 and the initial concentration, which gives the bracket. 0.10 mol of N₂O₄ in 1.0 L dissociates: N₂O₄ ⇌ 2 NO₂, Kc = 4.6 × 10⁻³. With x the amount dissociated, $K_c = (2x)^2 / (0.10 - x)$.

```{code-cell} ipython3
Kc, C0 = 4.6e-3, 0.10
f = lambda x: (2 * x) ** 2 / (C0 - x) - Kc
x = optimize.brentq(f, 1e-12, C0 - 1e-12)
print(f"x = {x:.4f} M, [N2O4] = {C0 - x:.4f} M, [NO2] = {2 * x:.4f} M, α = {x / C0:.1%}")
```

## Optimisation: finding minima

**Optimisation** finds the input that makes a function as small as possible (to maximise, minimise the negative). In chemistry the function is often an energy, and its minimum is the stable geometry; it is also the core of every fitting and machine-learning method. `optimize.minimize_scalar` handles a function of one variable, and `optimize.minimize` handles many variables at once, starting from an initial guess and following the slope downhill. The Lennard-Jones potential describes the interaction between two noble-gas atoms: strong repulsion at short range ($r^{-12}$), weak attraction at long range ($r^{-6}$), and a minimum in between whose position and depth we can find numerically and check against the exact answer $r = 2^{1/6}\sigma$.

```{code-cell} ipython3
eps, sigma = 0.997, 3.40          # kJ/mol, Å
LJ = lambda r: 4 * eps * ((sigma / r) ** 12 - (sigma / r) ** 6)
res = optimize.minimize_scalar(LJ, bounds=(3, 6), method="bounded")
print(f"r_min = {res.x:.3f} Å (theory 2^(1/6)σ = {2 ** (1/6) * sigma:.3f}), E_min = {res.fun:.3f} kJ/mol")
```

A multidimensional example, the geometry of a three-atom Lennard-Jones cluster. The nine variables are the x, y, z coordinates of the three atoms, flattened into one array because `minimize` expects a single vector. The BFGS algorithm estimates the curvature of the energy surface as it goes, which makes it efficient for smooth problems like this one.

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

Instrument signals contain noise, and the task is to reduce it without distorting the peaks. A simple moving average blurs sharp peaks, so analytical chemists use the **Savitzky–Golay filter**, which fits a small polynomial to each window of points and keeps peak heights and widths much better. `signal.savgol_filter(y, window_length, polyorder)` applies it; a wider window smooths more. `signal.find_peaks` then locates local maxima, and the `height` and `prominence` (how far a peak stands above its surroundings) options reject small noise bumps. A tiny example finds the peaks of a short list:

```{code-cell} ipython3
print(signal.find_peaks([0, 2, 1, 0, 3, 1, 0])[0])     # positions of the two maxima
```

The simulated HPLC chromatogram below has five compounds, two of which overlap, plus a drifting baseline and noise.

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

Reference data are tabulated at fixed intervals, but you often need a value in between: the density of water at 37 °C from a table at every 20 °C. **Interpolation** estimates values between known points. Linear interpolation joins points with straight lines (`np.interp`); a **cubic spline** joins them with smooth cubic curves whose slopes match at each point, which is more accurate for smooth physical properties. Interpolation is safe only *between* tabulated points; outside them (extrapolation) the estimate can be badly wrong.

```{code-cell} ipython3
print(np.interp(37, [20, 40], [0.99821, 0.99222]))    # straight line between two points
```

```{code-cell} ipython3
T_tab = np.array([0, 20, 40, 60, 80, 100])              # °C
rho_tab = np.array([0.99984, 0.99821, 0.99222, 0.98320, 0.97179, 0.95835])  # g/mL water
spl = interpolate.CubicSpline(T_tab, rho_tab)
print(f"density at 37 °C ≈ {spl(37):.5f} g/mL")
```

## Statistics: is the difference real?

Two analytical methods give slightly different mean results. Is the difference real, or could it be due to random scatter? A **t-test** answers this by comparing the difference between the means with the spread within each set. It returns a **p-value**, the probability of seeing a difference at least this large if the methods were really the same; a p-value below 0.05 is conventionally called *significant*. `stats.ttest_ind` compares two independent sets of measurements. `stats.linregress` fits a straight line and reports the slope, intercept and correlation coefficient $r$, the standard check of a calibration line.

```{code-cell} ipython3
method_A = np.array([10.12, 10.08, 10.15, 10.11, 10.09])   # % Fe
method_B = np.array([10.21, 10.18, 10.25, 10.19, 10.23])
tstat, pval = stats.ttest_ind(method_A, method_B)
print(f"t = {tstat:.2f}, p = {pval:.4f} ->", "significant" if pval < 0.05 else "not significant")
lin = stats.linregress([1, 2, 3, 4, 5], [0.101, 0.200, 0.302, 0.401, 0.499])
print(f"calibration slope {lin.slope:.4f}, R² = {lin.rvalue**2:.5f}")
```

```{admonition} Summary
:class: tip
- `scipy.constants` provides CODATA constants in SI units.
- `optimize.curve_fit` fits any model; `np.sqrt(np.diag(pcov))` gives parameter uncertainties. Give sensible starting guesses.
- `integrate.trapezoid`/`simpson` integrate data; `integrate.quad` integrates a function.
- `optimize.brentq` finds a bracketed root; `minimize_scalar` and `minimize` find minima.
- `signal.savgol_filter` smooths without flattening peaks; `signal.find_peaks` locates them.
- `interpolate.CubicSpline` estimates values between tabulated points (not outside them).
- `stats.ttest_ind` tests whether two means differ; `stats.linregress` checks a calibration.
```

## Exercises

**★ 1.** Use `optimize.brentq` to find the pH of 0.010 M acetic acid exactly (no approximation).

**★★ 2.** Fit the integrated second-order rate law $1/[A] = 1/[A]_0 + kt$ to simulated data and report k with its uncertainty.

**★★★ 3.** Find the global minimum geometry of a 7-atom Lennard-Jones cluster using `optimize.basinhopping`.
