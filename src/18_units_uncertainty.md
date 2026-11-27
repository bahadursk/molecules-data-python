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

A number without a unit is meaningless: "the volume is 22.4" could be litres, millilitres or cubic metres, and confusing them is one of the most common sources of wrong answers in chemistry, from student reports to lost spacecraft. A measurement without an uncertainty is incomplete: 0.1000 M and 0.10 ± 0.02 M say very different things about how far a result can be trusted. In ordinary Python code both kinds of information are lost, because a float is just a number. Two small libraries restore them. **pint** attaches units to numbers, converts between them and refuses to add quantities whose dimensions do not match. **uncertainties** attaches a standard uncertainty to each number and propagates it through every calculation automatically. This chapter shows how to use each, how to check the result by simulation, and how to report the final number sensibly.

## Unit-aware calculations with pint

pint works through a **unit registry**, an object that knows thousands of units and how they relate. You multiply a number by a unit from the registry to create a **quantity**, an object with both a magnitude and a unit. Arithmetic on quantities combines the units as well as the numbers, and `.to(unit)` converts the result into the unit you want. The basic idea fits in two lines:

```{code-cell} ipython3
import pint
ureg = pint.UnitRegistry()

d = 12.5 * ureg.mm
print(d, "=", d.to(ureg.cm), "=", d.to(ureg.inch))
```

Temperatures are a special case, because degrees Celsius are offset from kelvin rather than simply scaled. pint handles this correctly, but you must create Celsius values with `Quantity` (here abbreviated `Q_`) rather than by multiplication. With the ideal gas law, the units of $nRT/P$ cancel to a volume, and pint converts it to litres:

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

Because every quantity carries its **dimensions** (amount, mass, length, time, …), pint catches mistakes that a calculator never would. Adding moles to grams has no meaning, and instead of a number you get an error:

```{code-cell} ipython3
try:
    (2 * ureg.mol) + (3 * ureg.g)
except pint.DimensionalityError as e:
    print("Error:", e)
```

### Conversions

Chemistry uses an unusual variety of energy and wavelength units: electronvolts per molecule, kilojoules per mole, wavenumbers in cm⁻¹, hartrees from quantum chemistry. Some conversions are simple scalings; others, such as wavelength to photon energy, depend on physics ($E = hc/\lambda$). pint allows those through named **contexts**: `"spectroscopy"` (or `"sp"`) enables conversions between wavelength, frequency, wavenumber and energy. Multiplying by `avogadro_constant` turns a per-molecule energy into a per-mole one.

```{code-cell} ipython3
E = 1 * ureg.eV
print((E * ureg.avogadro_constant).to(ureg.kJ / ureg.mol))       # per mole
print((500 * ureg.nm).to(ureg.eV, "spectroscopy"))
print((1715 / ureg.cm).to(ureg.um ** -1), (1715 / ureg.cm).to(ureg.THz, "spectroscopy"))
print((1 * ureg.hartree * ureg.avogadro_constant).to(ureg.kcal / ureg.mol))
```

### Concentration and stoichiometry

Stoichiometry is where unit bookkeeping pays off most: mass divided by molar mass gives moles, and moles divided by volume gives concentration. pint tracks the combined unit (g ÷ (g/mol) ÷ mL) and converts it to mol/L, so a forgotten factor of 1000 is impossible. The format code `~P` prints the unit in short, pretty form.

```{code-cell} ipython3
M_NaCl = 58.44 * ureg.g / ureg.mol
mass = 2.922 * ureg.g
vol = 250.0 * ureg.mL
conc = (mass / M_NaCl / vol).to(ureg.mol / ureg.L)
print(f"{conc:.4f~P}")
```

### Arrays with units

A quantity's magnitude can be a whole NumPy array, so a column of data carries a single unit and converts in one step. `.magnitude` (or `.m`) returns the bare numbers when you need to pass them to a library that does not understand units.

```{code-cell} ipython3
import numpy as np
temps = Q_(np.array([0, 25, 100]), ureg.degC)
print(temps.to(ureg.K))
print(temps.to(ureg.K).magnitude)
```

## Uncertainty propagation

Every measured value has an uncertainty, often written as a standard deviation $\sigma$: a balance might read 0.2501 ± 0.0002 g. When you combine measured values, their uncertainties combine too. For $f(x, y)$ with independent errors:

$$\sigma_f^2 = \left(\frac{\partial f}{\partial x}\right)^2 \sigma_x^2 + \left(\frac{\partial f}{\partial y}\right)^2 \sigma_y^2$$

In words: each input's uncertainty contributes in proportion to how sensitive the result is to that input (the partial derivative), and independent contributions add in quadrature. For simple cases this gives rules you may have learned: for a sum, absolute uncertainties add in quadrature; for a product or quotient, *relative* uncertainties add in quadrature. Doing this by hand is tedious and error-prone. The `uncertainties` package does the differentiation for you. `ufloat(value, sigma)` creates an uncertain number, and every calculation with it carries the uncertainty along.

The most basic example adds two volumes delivered from a burette:

```{code-cell} ipython3
from uncertainties import ufloat, umath

v1 = ufloat(12.40, 0.02)          # mL
v2 = ufloat(10.15, 0.02)          # mL
print(v1 + v2)                    # uncertainty is sqrt(0.02² + 0.02²) ≈ 0.028
```

A standard solution of potassium hydrogen phthalate (KHP) combines a mass, a molar mass and a volume, and its concentration inherits all three uncertainties. The format `.2uP` rounds the uncertainty to two significant figures and the value to match.

```{code-cell} ipython3
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

For the Beer–Lambert law, $c = A/(\varepsilon l)$, the uncertainty in the molar absorptivity $\varepsilon$ is about 2 %, much larger than the relative uncertainty in $A$ or $l$. An **error budget** breaks the final uncertainty into the contributions from each input, which tells you where to spend effort: improving a measurement that contributes little is wasted work. `error_components()` returns each input's contribution. Labelling each input with a `tag` makes the budget readable:

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

Mathematical functions of uncertain numbers must come from `umath` (or NumPy-compatible versions in `unumpy`), not from `math`, because the ordinary `math.log` only knows how to handle plain floats. The pH of a solution and an activation energy from two rate constants (the two-point Arrhenius equation) both propagate their uncertainties correctly:

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

The propagation formula above is a *linear* approximation: it assumes the function is close to a straight line over the range of each input's uncertainty. For strongly non-linear functions, such as exponentials or ratios with large relative errors, the approximation can mislead. **Monte Carlo** propagation avoids it by brute force. You draw many random samples of each input from its distribution, compute the result for every sample, and look at the spread of the results. It needs no derivatives, works for any function, and reveals whether the result's distribution is skewed. The price is computation, which is cheap with NumPy arrays.

A basic Monte Carlo run checks the burette sum from above:

```{code-cell} ipython3
import numpy as np
rng = np.random.default_rng(0)
total = rng.normal(12.40, 0.02, 100_000) + rng.normal(10.15, 0.02, 100_000)
print(f"{total.mean():.3f} ± {total.std():.3f} mL")
```

The two methods agree, as they should for a sum. For the activation energy the agreement is also close, and the histogram shows the full distribution of possible values:

```{code-cell} ipython3
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

A result should be reported with as many digits as its uncertainty justifies, and no more. The usual convention is to round the uncertainty to one or two significant figures (two if its first digit is 1, so that 0.14 is not rounded to 0.1, a 30 % change) and then round the value to the same decimal place. Writing 0.0293547 ± 0.000672 implies a precision that does not exist; 0.0294 ± 0.0007 says the same thing honestly. The helper below applies this rule.

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

```{admonition} Summary
:class: tip
- pint: create a registry, multiply numbers by units, convert with `.to()`; use `Q_` for Celsius; dimension mismatches raise errors.
- The `"spectroscopy"` context converts between wavelength, frequency, wavenumber and energy.
- uncertainties: `ufloat(value, sigma)` propagates standard uncertainties through any calculation; use `umath` for functions.
- `error_components()` gives the error budget, showing which input to improve.
- Monte Carlo propagation samples inputs and needs no linear approximation.
- Report the uncertainty to 1–2 significant figures and round the value to match.
```

## Exercises

**★ 1.** Use pint to convert a 15.0 psi tyre pressure to kPa and bar.

**★★ 2.** A titration uses 24.35 ± 0.05 mL of 0.1000 ± 0.0002 M NaOH to neutralise 25.00 ± 0.03 mL of HCl. Find [HCl] with its uncertainty.

**★★★ 3.** Compare linear and Monte Carlo uncertainty for $K = e^{-\Delta G/RT}$ with ΔG = −20 ± 2 kJ/mol at 298 K. Why do they differ?
