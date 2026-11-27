---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Writing Your Own Functions

```{admonition} 🟢 Beginner · school & first-year
:class: level-1
Builds on Chapters 1–5.
```

A **function** packages a calculation so you can reuse it, test it and give it a meaningful name. Once you have written `molar_mass("C6H12O6")` you never need to add up atomic masses by hand again.

## Defining a function

```{code-cell} ipython3
def celsius_to_kelvin(t_c):
    return t_c + 273.15

print(celsius_to_kelvin(25))
print(celsius_to_kelvin(-196))   # liquid nitrogen
```

- `def` starts the definition, followed by the name and the **parameters** in brackets.
- The indented body runs each time the function is **called**.
- `return` sends a value back to the caller.

## Several parameters, defaults and keywords

```{code-cell} ipython3
def ideal_gas_volume(n, T, P=101_325, R=8.314462618):
    """Volume in m^3 of n mol of ideal gas at T (K) and P (Pa)."""
    return n * R * T / P

print(ideal_gas_volume(1, 273.15))                # default pressure
print(ideal_gas_volume(1, 273.15, P=200_000))     # keyword argument
print(ideal_gas_volume(T=300, n=2))               # any order with keywords
```

## Docstrings and type hints

A **docstring** describes what a function does. **Type hints** say what kinds of values it expects. Both help you and your readers, and `help()` displays them.

```{code-cell} ipython3
def dilution(c1: float, v1: float, v2: float) -> float:
    """Final concentration after diluting volume v1 at concentration c1 to v2.

    Uses c1 * v1 = c2 * v2. Volumes can be in any unit, as long as both match.
    """
    return c1 * v1 / v2

help(dilution)
print(dilution(1.0, 10, 250), "mol/L")
```

## Returning several values

```{code-cell} ipython3
def quadratic_roots(a, b, c):
    disc = (b**2 - 4*a*c) ** 0.5
    return (-b + disc) / (2*a), (-b - disc) / (2*a)

# Weak-acid equilibrium: x^2 + Ka x - Ka C = 0  for 0.10 M acetic acid
Ka, C = 1.8e-5, 0.10
x1, x2 = quadratic_roots(1, Ka, -Ka * C)
import math
print(f"[H3O+] = {x1:.3e} M, pH = {-math.log10(x1):.2f}  (the other root, {x2:.1e}, is unphysical)")
```

## Scope

Variables created inside a function are **local**: they disappear when the function ends.

```{code-cell} ipython3
R = 8.314
def energy(T):
    kT = R * T       # local variable, R is read from outside
    return kT

print(energy(298))
print("kT" in globals())
```

```{tip}
Pass everything a function needs as parameters rather than relying on outside variables. Functions that only use their inputs are easy to test and reuse.
```

## The molar-mass function

Let us bring together strings, dictionaries, loops and functions.

```{code-cell} ipython3
import re

ATOMIC_MASS = {"H": 1.008, "He": 4.0026, "Li": 6.94, "B": 10.81, "C": 12.011, "N": 14.007,
               "O": 15.999, "F": 18.998, "Na": 22.990, "Mg": 24.305, "Al": 26.982, "Si": 28.085,
               "P": 30.974, "S": 32.06, "Cl": 35.45, "K": 39.098, "Ca": 40.078, "Fe": 55.845,
               "Cu": 63.546, "Zn": 65.38, "Br": 79.904, "Ag": 107.87, "I": 126.90}

def count_atoms(formula: str) -> dict:
    """Count atoms in a formula, supporting brackets, e.g. 'Ca3(PO4)2'."""
    tokens = re.findall(r"[A-Z][a-z]?|\(|\)|\d+", formula)
    stack = [{}]                       # one dictionary per open bracket
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        i += 1
        if tok == "(":
            stack.append({})
            continue
        mult = 1
        if i < len(tokens) and tokens[i].isdigit():
            mult = int(tokens[i])
            i += 1
        if tok == ")":
            group = stack.pop()
            for el, n in group.items():
                stack[-1][el] = stack[-1].get(el, 0) + n * mult
        else:
            stack[-1][tok] = stack[-1].get(tok, 0) + mult
    return stack[0]

def molar_mass(formula: str) -> float:
    """Molar mass in g/mol."""
    return sum(ATOMIC_MASS[el] * n for el, n in count_atoms(formula).items())

for f in ["H2O", "C6H12O6", "Ca3(PO4)2", "CuSO4", "Al2(SO4)3", "C8H10N4O2"]:
    print(f"{f:10s} {count_atoms(f)}  M = {molar_mass(f):8.3f} g/mol")
```

## Mass percent composition

Functions can call other functions:

```{code-cell} ipython3
def mass_percent(formula):
    M = molar_mass(formula)
    return {el: 100 * ATOMIC_MASS[el] * n / M for el, n in count_atoms(formula).items()}

import matplotlib.pyplot as plt
comp = mass_percent("C8H10N4O2")
palette = {"C": "#444444", "H": "#cfd8dc", "N": "#3f51b5", "O": "#e53935"}   # CPK-like colours
plt.pie(comp.values(), labels=[f"{k} {v:.1f}%" for k, v in comp.items()],
        colors=[palette[k] for k in comp], wedgeprops=dict(width=0.45, edgecolor="white"))
plt.title("Mass composition of caffeine")
plt.show()
```

## Small anonymous functions: `lambda`

`lambda` creates a one-line function, often used as a sort key.

```{code-cell} ipython3
acids = [("HCl", -6.3), ("acetic", 4.76), ("HF", 3.17), ("phenol", 9.95), ("formic", 3.75)]
for name, pKa in sorted(acids, key=lambda a: a[1]):
    print(f"{name:8s} pKa {pKa:5.2f}")

strong = list(filter(lambda a: a[1] < 0, acids))
Kas = list(map(lambda a: 10 ** -a[1], acids))
print(strong, [f"{K:.1e}" for K in Kas])
```

## Exercises

**★ 1.** Write `kelvin_to_fahrenheit(T)`.

**★★ 2.** Write `moles_from_mass(mass, formula)` using `molar_mass`. How many moles are in 5.00 g of CuSO₄·5H₂O? (Hint: CuSO4H10O5.)

```{admonition} Solution
:class: dropdown
`5.00 / molar_mass("CuSO9H10")` ≈ 0.0200 mol (M = 249.7 g/mol).
```

**★★★ 3.** Write `empirical_formula(percentages)` that turns mass percentages, e.g. `{"C": 40.0, "H": 6.7, "O": 53.3}`, into the empirical formula (`CH2O`).
