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

A **function** is a named block of code that takes some inputs, does a job and hands back a result. You have already used many functions that come with Python, such as `print()`, `len()` and `round()`. Writing your own is the single most important step from typing calculations to writing programs. A function lets you write a calculation once and reuse it hundreds of times, gives it a meaningful name so that code reads like the science it describes, and lets you test that one piece in isolation. Once you have written `molar_mass("C6H12O6")` you never need to add up atomic masses by hand again, and if you find a mistake you fix it in one place. This chapter shows how to define functions, pass information into them, get results out, and combine small functions into a useful chemistry tool.

## Defining a function

A function definition starts with the keyword `def`, followed by the function's name, a pair of parentheses containing the names of its inputs, and a colon. The inputs are called **parameters**. The indented lines below form the **body**, the code that runs each time the function is used. The keyword `return` ends the function and sends a value back to whoever called it. Defining a function does not run it; it only teaches Python a new word. The body runs when you **call** the function by writing its name followed by parentheses containing actual values, called **arguments**.

The most basic function takes one number and returns another:

```{code-cell} ipython3
def celsius_to_kelvin(t_c):
    return t_c + 273.15

print(celsius_to_kelvin(25))
print(celsius_to_kelvin(-196))   # liquid nitrogen
```

When Python runs `celsius_to_kelvin(25)`, it sets the parameter `t_c` to 25, runs the body, and the whole call is replaced by the returned value 298.15. That value can be printed, stored in a variable or used in further arithmetic, just like any other number.

```{code-cell} ipython3
T = celsius_to_kelvin(37)          # body temperature
print(T, T * 8.314)                # use the result in a further calculation
```

A function without a `return` statement still runs, but it gives back the special value `None`. Forgetting `return` is a common beginner's bug: the function seems to work because it prints something, yet its result cannot be used.

```{code-cell} ipython3
def show_kelvin(t_c):
    print(t_c + 273.15)            # prints, but returns nothing

result = show_kelvin(25)
print("result is", result)
```

## Several parameters, defaults and keywords

A function can take as many parameters as it needs, separated by commas. When you call it, arguments are matched to parameters by position: the first argument goes to the first parameter, and so on. A parameter can be given a **default value** with `=` in the definition, which makes that argument optional; the default is used whenever the caller leaves it out. Defaults are ideal for values that are usually the same, such as standard pressure or the gas constant.

```{code-cell} ipython3
def concentration(n, V):
    """Concentration in mol/L from moles n and volume V in L."""
    return n / V

print(concentration(0.050, 0.250))
```

You can also pass an argument by name, as `P=200_000`. These **keyword arguments** can be given in any order and make the call self-explanatory, which matters when a function has several numeric parameters that are easy to mix up. The underscores in `101_325` are only visual separators and are ignored by Python.

```{code-cell} ipython3
def ideal_gas_volume(n, T, P=101_325, R=8.314462618):
    """Volume in m^3 of n mol of ideal gas at T (K) and P (Pa)."""
    return n * R * T / P

print(ideal_gas_volume(1, 273.15))                # default pressure
print(ideal_gas_volume(1, 273.15, P=200_000))     # keyword argument
print(ideal_gas_volume(T=300, n=2))               # any order with keywords
```

The first result, 0.0224 m³, is the familiar 22.4 L molar volume at 0 °C and 1 atm, a good check that the function is right.

## Docstrings and type hints

A **docstring** is a string placed on the first line of a function's body that explains what the function does, what its inputs mean and in which units. It is not a comment: Python stores it with the function, and `help()` displays it. **Type hints** are optional notes after each parameter (`c1: float`) and after the parentheses (`-> float`) saying what kinds of values the function expects and returns. Python does not enforce them, but they tell readers, and editors that check code, how the function is meant to be used. Both are cheap to write and save you time when you return to your code months later.

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

A function can return more than one value by separating them with commas after `return`. Python packs them into a tuple, and the caller can unpack the tuple straight into separate names (Chapter 4). This is the natural way to return results that come in groups, such as a mean and a standard deviation, or the two roots of a quadratic.

```{code-cell} ipython3
def mean_and_range(values):
    return sum(values) / len(values), max(values) - min(values)

avg, spread = mean_and_range([25.10, 25.15, 25.05])
print(f"mean {avg:.3f} mL, range {spread:.2f} mL")
```

Weak-acid equilibria lead to a quadratic. For acetic acid at concentration $C$, the hydronium concentration $x$ satisfies $x^2 + K_a x - K_a C = 0$, and only the positive root is physical:

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

**Scope** is the region of a program in which a variable name is visible. Variables created inside a function, including its parameters, are **local**: they exist only while the function runs and vanish when it returns. This is a feature, not a limitation. It means a function's internal workings cannot accidentally overwrite a variable of the same name elsewhere in your notebook, so you can use short names such as `x` or `n` inside functions without fear. A function *can* read variables defined outside it (global variables), but relying on that makes the function depend on hidden state.

```{code-cell} ipython3
R = 8.314
def energy(T):
    kT = R * T       # local variable, R is read from outside
    return kT

print(energy(298))
print("kT" in globals())
```

The last line asks whether `kT` exists at the top level of the notebook, and the answer is `False`: it lived only inside the call.

```{tip}
Pass everything a function needs as parameters rather than relying on outside variables. Functions that only use their inputs are easy to test and reuse.
```

## The molar-mass function

Real programs are built from small functions that each do one job, with larger functions calling smaller ones. Here strings, regular expressions, dictionaries, loops and functions come together. `count_atoms` turns a formula into a dictionary of atom counts and handles brackets by keeping a *stack*, a list with one dictionary per open bracket: an opening bracket starts a new dictionary, and a closing bracket multiplies that dictionary's counts by the number after it and merges them into the level below. `molar_mass` then needs only one line, because it reuses `count_atoms`.

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

`ATOMIC_MASS` is written in capitals by convention to signal a *constant*, a value set once and never changed. Because the two functions are separate, you can check `count_atoms` on its own before trusting `molar_mass`, which is exactly how you should build any larger calculation.

## Mass percent composition

Once a function exists, other functions can call it, and each new tool becomes easier to write. The mass percent of an element is its total mass in the formula divided by the molar mass, times 100. Writing `mass_percent` takes two lines because the hard parts are already done:

```{code-cell} ipython3
def mass_percent(formula):
    M = molar_mass(formula)
    return {el: 100 * ATOMIC_MASS[el] * n / M for el, n in count_atoms(formula).items()}

print(mass_percent("H2O"))
```

A ring chart makes the composition of caffeine easy to see, using colours close to the CPK colours chemists use for atoms:

```{code-cell} ipython3
import matplotlib.pyplot as plt
comp = mass_percent("C8H10N4O2")
palette = {"C": "#444444", "H": "#cfd8dc", "N": "#3f51b5", "O": "#e53935"}   # CPK-like colours
plt.pie(comp.values(), labels=[f"{k} {v:.1f}%" for k, v in comp.items()],
        colors=[palette[k] for k in comp], wedgeprops=dict(width=0.45, edgecolor="white"))
plt.title("Mass composition of caffeine")
plt.show()
```

## Small anonymous functions: `lambda`

Sometimes you need a tiny function for a single use, typically to tell another function how to do its job. Writing a full `def` for that is clumsy, so Python offers `lambda`, which creates a nameless one-line function: `lambda x: x * 2` takes `x` and returns `x * 2`. A lambda can contain only one expression and no statements, which keeps it short by design. Its most common use is as the `key` for `sorted`, which says what to sort by.

```{code-cell} ipython3
double = lambda x: x * 2
print(double(21))
```

Here a list of (name, pKa) tuples is sorted by the second item, making the strongest acid come first. `filter` keeps the items for which a function returns `True`, and `map` applies a function to every item; both are shown with lambdas, although the list comprehensions of Chapter 7 are often clearer.

```{code-cell} ipython3
acids = [("HCl", -6.3), ("acetic", 4.76), ("HF", 3.17), ("phenol", 9.95), ("formic", 3.75)]
for name, pKa in sorted(acids, key=lambda a: a[1]):
    print(f"{name:8s} pKa {pKa:5.2f}")

strong = list(filter(lambda a: a[1] < 0, acids))
Kas = list(map(lambda a: 10 ** -a[1], acids))
print(strong, [f"{K:.1e}" for K in Kas])
```

```{admonition} Summary
:class: tip
- `def name(parameters):` defines a function; calling `name(arguments)` runs its body, and `return` sends back the result.
- A function with no `return` gives `None`.
- Defaults (`P=101_325`) make arguments optional; keyword arguments (`P=2e5`) make calls clear.
- Docstrings and type hints document what a function expects and returns; `help()` shows them.
- `return a, b` returns a tuple that the caller can unpack.
- Variables inside a function are local; pass inputs as parameters.
- Build big tools from small functions that call each other; use `lambda` for throwaway one-liners.
```

## Exercises

**★ 1.** Write `kelvin_to_fahrenheit(T)`.

```{admonition} Solution
:class: dropdown
`def kelvin_to_fahrenheit(T): return (T - 273.15) * 9 / 5 + 32`; `kelvin_to_fahrenheit(373.15)` gives 212.0.
```

**★★ 2.** Write `moles_from_mass(mass, formula)` using `molar_mass`. How many moles are in 5.00 g of CuSO₄·5H₂O? (Hint: CuSO4H10O5.)

```{admonition} Solution
:class: dropdown
`5.00 / molar_mass("CuSO4H10O5")` ≈ 0.0200 mol (M = 249.7 g/mol).
```

**★★★ 3.** Write `empirical_formula(percentages)` that turns mass percentages, e.g. `{"C": 40.0, "H": 6.7, "O": 53.3}`, into the empirical formula (`CH2O`).
