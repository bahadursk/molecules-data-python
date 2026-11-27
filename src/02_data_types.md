---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Numbers, Booleans and Types

```{admonition} 🟢 Beginner · school & first-year
:class: level-1
Builds on Chapter 1.
```

Every value in Python has a **type**. The type decides what you can do with it: you can divide two masses, but not a mass by the word "carbon".

## The built-in numeric types

```{code-cell} ipython3
n_atoms = 6              # int: whole numbers, counts of things
mass = 12.011            # float: real numbers with a decimal point
wave = 3 + 4j            # complex: used in quantum mechanics and Fourier transforms
is_metal = False         # bool: True or False

for value in (n_atoms, mass, wave, is_metal):
    print(repr(value), "is", type(value).__name__)
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Use `int` for things you count (atoms, electrons, protons) and `float` for things you measure (masses, energies, concentrations). Counting 6.0 electrons does not make sense; measuring exactly 12 g does not happen.
```

## Converting between types

```{code-cell} ipython3
reading = "0.0253"               # text typed by a user or read from a file
conc = float(reading)            # now a number
print(conc * 1000, "mmol/L")

print(int(7.9))                  # int() truncates towards zero
print(round(7.9), round(2.5), round(3.5))   # round() uses banker's rounding
print(str(18.015) + " g/mol")
```

## Integer and float division

```{code-cell} ipython3
electrons = 17
print(electrons / 2)    # always a float
print(electrons // 2)   # electron pairs
print(electrons % 2)    # unpaired electrons left over
```

The `%` operator answers a chemistry question directly: an odd electron count means a **radical**.

## Floating-point precision

Computers store floats in binary, and most decimals cannot be represented exactly.

```{code-cell} ipython3
print(0.1 + 0.2)
print(0.1 + 0.2 == 0.3)
```

Never test floats with `==`. Use `math.isclose` instead.

```{code-cell} ipython3
import math
print(math.isclose(0.1 + 0.2, 0.3))

measured, literature = 1.0012, 1.0010     # g/mL
print(math.isclose(measured, literature, rel_tol=1e-3))   # within 0.1 %?
```

## Comparisons and Booleans

Comparisons produce `True` or `False`.

| Operator | Meaning |
|---|---|
| `==`, `!=` | equal, not equal |
| `<`, `<=`, `>`, `>=` | ordering |
| `and`, `or`, `not` | combine Booleans |

```{code-cell} ipython3
pH = 4.2
print("acidic:", pH < 7)
print("weakly acidic:", 4 < pH < 7)    # chained comparison

boiling_point = 78.4   # °C, ethanol
room_temp = 25
print("liquid at room temperature:", room_temp < boiling_point and room_temp > -114)
```

## Complex numbers

Complex numbers appear in wavefunctions and in the Fourier transform used by every NMR spectrometer.

```{code-cell} ipython3
import cmath
psi = cmath.exp(1j * math.pi / 3)      # a phase factor e^{iθ}
print(psi)
print("probability |ψ|² =", abs(psi) ** 2)
```

## Very large and very small numbers

```{code-cell} ipython3
print(2 ** 200)                 # ints have unlimited size
print(1e308 * 10)               # floats overflow to infinity
print(5e-324 / 2)               # and underflow to zero
```

```{tip}
Equilibrium constants can be enormous (K ≈ 10⁸⁰). Work with $\log K$ or $\Delta G$ instead to stay within float range.
```

## Exercises

**★ 1.** An element has 26 protons and forms an ion with 23 electrons. Use integers to compute its charge and print `Fe3+`.

```{admonition} Solution
:class: dropdown
`charge = 26 - 23; print(f"Fe{charge}+")`
```

**★★ 2.** Write a Boolean expression that is `True` if a temperature `T` in kelvin lies between the melting point (273.15 K) and boiling point (373.15 K) of water.

```{admonition} Solution
:class: dropdown
`273.15 < T < 373.15`
```

**★★★ 3.** Compute $K = e^{-\Delta G / RT}$ for ΔG = −500 kJ mol⁻¹ at 298 K. What happens? Compute $\log_{10} K$ instead.

```{admonition} Solution
:class: dropdown
`math.exp(500e3/(8.314*298))` raises `OverflowError`. `500e3/(8.314*298)/math.log(10)` gives log K ≈ 87.6.
```
