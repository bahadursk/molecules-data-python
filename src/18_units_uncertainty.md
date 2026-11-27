---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Units and Uncertainty: pint and uncertainties

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Part I.
```

A number without a unit is meaningless, and a measurement without an uncertainty is incomplete. Two small libraries enforce good habits: **pint** attaches units to numbers, and **uncertainties** propagates errors automatically.

## Unit-aware calculations with pint

```{code-cell} ipython3
import pint
ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

n = 2.50 * ureg.mol
T = Q_(25, ureg.degC).to(ureg.kelvin)
P = 1.00 * ureg.atm
R = 8.314462618 * ureg.J / (ureg.mol * ureg.K)

V = (n * R * T / P).to(ureg.L)
print(V, "|", f"{V:.2f~P}")
```

pint catches mistakes that a calculator never would:

```{code-cell} ipython3
try:
    (2 * ureg.mol) + (3 * ureg.g)
except pint.DimensionalityError as e:
    print("Error:", e)
```

### Conversions

```{code-cell} ipython3
E = 1 * ureg.eV
print((E * ureg.avogadro_constant).to(ureg.kJ / ureg.mol))       # per mole
print((500 * ureg.nm).to(ureg.eV, "spectroscopy"))
print((1715 / ureg.cm).to(ureg.um ** -1), (1715 / ureg.cm).to(ureg.THz, "spectroscopy"))
print((1 * ureg.hartree * ureg.avogadro_constant).to(ureg.kcal / ureg.mol))
```

### Concentration and stoichiometry

```{code-cell} ipython3
M_NaCl = 58.44 * ureg.g / ureg.mol
mass = 2.922 * ureg.g
vol = 250.0 * ureg.mL
conc = (mass / M_NaCl / vol).to(ureg.mol / ureg.L)
print(f"{conc:.4f~P}")
```

### Arrays with units

```{code-cell} ipython3
import numpy as np
temps = Q_(np.array([0, 25, 100]), ureg.degC)
print(temps.to(ureg.K))
```

## Uncertainty propagation

When you combine measured values, their uncertainties combine too. For $f(x, y)$ with independent errors:

$$\sigma_f^2 = \left(\frac{\partial f}{\partial x}\right)^2 \sigma_x^2 + \left(\frac{\partial f}{\partial y}\right)^2 \sigma_y^2$$

The `uncertainties` package does this differentiation for you.

```{code-cell} ipython3
from uncertainties import ufloat, umath

mass = ufloat(0.2501, 0.0002)      # g, analytical balance
volume = ufloat(25.00, 0.03)       # mL, volumetric flask
M = ufloat(204.22, 0.01)           # g/mol, KHP
conc = mass / M / volume * 1000
print(f"c(KHP) = {conc:.2uP} mol/L")   # 2 significant digits on the uncertainty
```

```{code-cell} ipython3
A = ufloat(0.452, 0.004)            # absorbance
eps = ufloat(15400, 300)            # L/(mol cm)
l = ufloat(1.000, 0.002)            # cm
c = A / (eps * l)
print(f"c = {c:.2uS} mol/L")
print("contributions:", {str(k.tag) if k.tag else "?": round(v, 9) for k, v in c.error_components().items()})
```

Label each input to see which one dominates the error budget:

```{code-cell} ipython3
A = ufloat(0.452, 0.004, tag="absorbance")
eps = ufloat(15400, 300, tag="epsilon")
l = ufloat(1.000, 0.002, tag="path")
c = A / (eps * l)

import matplotlib.pyplot as plt
comp = {k.tag: (v / c.s) ** 2 for k, v in c.error_components().items()}
plt.barh(list(comp), list(comp.values()), color=["#0f9d8a", "#e5487a", "#f2a93b"])
plt.xlabel("fraction of variance"); plt.title("Error budget: ε dominates")
plt.show()
```

### Functions of uncertain numbers

```{code-cell} ipython3
H = ufloat(3.2e-4, 0.1e-4)
pH = -umath.log10(H)
print(f"pH = {pH:.2uP}")

k298 = ufloat(1.2e-3, 0.1e-3)
k318 = ufloat(6.5e-3, 0.4e-3)
Ea = 8.314 * umath.log(k318 / k298) / (1 / 298 - 1 / 318) / 1000
print(f"Ea = {Ea:.2uP} kJ/mol")
```

## Monte Carlo error propagation

For strongly non-linear functions, linear propagation can mislead. Sampling is the robust alternative.

```{code-cell} ipython3
rng = np.random.default_rng(0)
N = 100_000
k298_s = rng.normal(1.2e-3, 0.1e-3, N)
k318_s = rng.normal(6.5e-3, 0.4e-3, N)
Ea_s = 8.314 * np.log(k318_s / k298_s) / (1 / 298 - 1 / 318) / 1000
print(f"Monte Carlo Ea = {Ea_s.mean():.1f} ± {Ea_s.std():.1f} kJ/mol")
plt.hist(Ea_s, bins=80, color="#6c4ad6", alpha=0.8)
plt.axvline(Ea.n, color="k", ls="--", label="linear propagation")
plt.xlabel("Ea / kJ mol$^{-1}$"); plt.legend(); plt.show()
```

## Significant figures in reports

```{code-cell} ipython3
def report(value, sigma):
    """Round sigma to 1–2 significant figures and the value to match."""
    from math import floor, log10
    digits = -int(floor(log10(sigma)))
    if round(sigma, digits) < 2 * 10 ** -digits:
        digits += 1
    return f"{round(value, digits)} ± {round(sigma, digits)}"

print(report(0.0293547, 0.000672), "|", report(2045.37, 13.2))
```

## Exercises

**★ 1.** Use pint to convert a 15.0 psi tyre pressure to kPa and bar.

**★★ 2.** A titration uses 24.35 ± 0.05 mL of 0.1000 ± 0.0002 M NaOH to neutralise 25.00 ± 0.03 mL of HCl. Find [HCl] with its uncertainty.

**★★★ 3.** Compare linear and Monte Carlo uncertainty for $K = e^{-\Delta G/RT}$ with ΔG = −20 ± 2 kJ/mol at 298 K. Why do they differ?
