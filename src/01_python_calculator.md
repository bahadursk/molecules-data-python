---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Python as Your Chemistry Calculator

```{admonition} 🟢 Beginner · school & first-year
:class: level-1
No programming experience needed. You only need to know what a mole is.
```

Every chemist starts with a calculator. In this chapter Python *becomes* that calculator, one that remembers your numbers, never mistypes Avogadro's constant and can show its working.

## Running your first cell

A Jupyter notebook is a list of **cells**. Click a code cell and press **Shift + Enter** to run it. The result appears underneath.

```{code-cell} ipython3
print("Hello, chemistry!")
```

`print` is a **function**: a named piece of code that does a job. The text inside the quotation marks is a **string**.

## Arithmetic

Python understands the usual operators.

| Operator | Meaning | Example | Result |
|---|---|---|---|
| `+` | add | `12.011 + 2 * 15.999` | 44.009 |
| `-` | subtract | `100 - 37.5` | 62.5 |
| `*` | multiply | `2 * 1.008` | 2.016 |
| `/` | divide | `18.0 / 2` | 9.0 |
| `**` | power | `10 ** -3` | 0.001 |
| `//` | floor division | `17 // 5` | 3 |
| `%` | remainder | `17 % 5` | 2 |

Let us compute the molar mass of water, H₂O, using atomic masses in g mol⁻¹.

```{code-cell} ipython3
2 * 1.008 + 15.999
```

And carbon dioxide, CO₂:

```{code-cell} ipython3
12.011 + 2 * 15.999
```

```{warning}
Python follows the normal order of operations: brackets, powers, multiply and divide, then add and subtract. `2 * 1.008 + 15.999` is *not* the same as `2 * (1.008 + 15.999)`.
```

## Scientific notation

Chemists live with very large and very small numbers. Write them with `e`, which means "times ten to the power of".

```{code-cell} ipython3
avogadro = 6.02214076e23   # mol^-1
planck = 6.62607015e-34    # J s
avogadro * planck
```

## Variables: giving numbers names

A **variable** stores a value under a name, using `=`. Anything after `#` is a **comment** for humans and is ignored by Python.

```{code-cell} ipython3
m_H = 1.008    # g/mol
m_C = 12.011
m_O = 15.999

M_ethanol = 2 * m_C + 6 * m_H + m_O   # C2H5OH = C2H6O
M_ethanol
```

Now we can answer a real question: **how many moles are in 10.0 g of ethanol?**

$$n = \frac{m}{M}$$

```{code-cell} ipython3
mass = 10.0                 # g
moles = mass / M_ethanol
moles
```

And how many molecules is that?

```{code-cell} ipython3
molecules = moles * avogadro
molecules
```

```{tip}
Good variable names make code read like chemistry: `moles = mass / molar_mass` explains itself. Names may contain letters, digits and underscores, cannot start with a digit, and are case sensitive (`M` and `m` are different).
```

## The `math` module

Logarithms, square roots and exponentials live in the `math` module, which you **import** once.

```{code-cell} ipython3
import math

conc_H = 3.2e-4            # mol/L of H3O+
pH = -math.log10(conc_H)
pH
```

The Arrhenius equation, $k = A\,e^{-E_a/RT}$, for a reaction with $E_a = 50$ kJ mol⁻¹:

```{code-cell} ipython3
A = 1.0e13        # s^-1
Ea = 50_000       # J/mol  (underscores make long numbers readable)
R = 8.314         # J/(mol K)

for T in (298, 308, 318):
    k = A * math.exp(-Ea / (R * T))
    print(T, "K ->", k, "s^-1")
```

Notice that a 20 K rise roughly increases $k$ fourfold. We will learn what `for` does in Chapter 5.

## Presenting answers: f-strings and significant figures

Long rows of digits are not a result. An **f-string** places values inside text and controls how they are formatted.

```{code-cell} ipython3
print(f"Molar mass of ethanol: {M_ethanol:.2f} g/mol")
print(f"Moles in {mass} g: {moles:.3f} mol")
print(f"Molecules: {molecules:.3e}")
print(f"pH = {pH:.2f}")
```

| Format | Meaning | `1234.5678` becomes |
|---|---|---|
| `:.2f` | 2 decimal places | `1234.57` |
| `:.3e` | scientific, 3 decimals | `1.235e+03` |
| `:.3g` | 3 significant figures | `1.23e+03` |
| `:,.1f` | thousands separator | `1,234.6` |

## A worked example: the ideal gas law

What volume does 2.50 mol of N₂ occupy at 25 °C and 1.00 atm? Use $PV = nRT$.

```{code-cell} ipython3
n = 2.50                 # mol
T = 25 + 273.15          # K
P = 101_325              # Pa (1 atm)
R = 8.314462618          # J/(mol K)

V = n * R * T / P        # m^3
print(f"V = {V:.4f} m³ = {V * 1000:.1f} L")
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Always convert to SI units before using $R = 8.314$ J mol⁻¹ K⁻¹: pressure in pascals, volume in cubic metres, temperature in kelvin. In Chapter 18 the `pint` library will do this checking for you.
```

## Exercises

**★ 1.** Calculate the molar mass of glucose, C₆H₁₂O₆.

```{admonition} Solution
:class: dropdown
`6 * 12.011 + 12 * 1.008 + 6 * 15.999` gives 180.156 g/mol.
```

**★★ 2.** A solution of HCl has pH 2.75. Calculate [H₃O⁺] using `10 ** -pH`, and print it to 3 significant figures.

```{admonition} Solution
:class: dropdown
`print(f"{10 ** -2.75:.3g} mol/L")` prints `0.00178 mol/L`.
```

**★★★ 3.** Using the Arrhenius equation, find the temperature at which the rate constant of the reaction above reaches 1.0 s⁻¹. Rearrange to $T = E_a / (R \ln(A/k))$.

```{admonition} Solution
:class: dropdown
`Ea / (R * math.log(A / 1.0))` gives about 201 K.
```

```{admonition} 🤖 ML corner
:class: ml-corner
Machine-learning models are just functions that turn numbers into predictions, exactly like `moles = mass / M`. The difference is that the formula is *learned* from data. You will build your first one in Chapter 31.
```
