---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Errors, Exceptions and Debugging

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 1–8.
```

Every programmer spends a large part of their time fixing errors. The skill is not avoiding errors but **reading** them quickly. This chapter teaches you to understand a traceback, handle problems gracefully and write code that fails loudly when chemistry goes wrong.

## Reading a traceback

```{code-cell} ipython3
:tags: [raises-exception]
masses = {"H": 1.008, "O": 15.999}
M = 2 * masses["H"] + masses["0"]      # a zero instead of the letter O
```

Read a traceback **from the bottom up**: the last line gives the exception type (`KeyError`) and the problem (`'0'`); the lines above show where it happened.

## The errors you will meet most

| Exception | Typical cause | Chemistry example |
|---|---|---|
| `SyntaxError` | missing bracket, colon or quote | `print("pH"` |
| `NameError` | using a name before defining it | `molarmass` vs `molar_mass` |
| `TypeError` | wrong type of value | `"12.0" * 2.5` |
| `ValueError` | right type, impossible value | `float("n/a")`, `math.log(-1)` |
| `KeyError` | missing dictionary key | an element not in your table |
| `IndexError` | list position out of range | the 11th point of a 10-point spectrum |
| `ZeroDivisionError` | dividing by zero | a blank with zero absorbance |
| `FileNotFoundError` | wrong path | `"data/uvvis.csv"` vs `"data/uv_vis.csv"` |

```{code-cell} ipython3
:tags: [raises-exception]
import math
pH = -math.log10(0.0)
```

## Handling exceptions with `try` / `except`

```{code-cell} ipython3
raw_readings = ["0.512", "0.498", "n/a", "0.505", "", "0.511"]
values = []
for r in raw_readings:
    try:
        values.append(float(r))
    except ValueError:
        print(f"skipping unreadable value {r!r}")
print(values)
```

The full form has four parts:

```{code-cell} ipython3
def safe_pH(conc):
    try:
        pH = -math.log10(conc)
    except ValueError:
        print("concentration must be positive")
        return None
    else:                         # runs if no exception
        return round(pH, 2)
    finally:                      # always runs, useful for closing instruments
        pass

print(safe_pH(1e-3), safe_pH(-1))
```

```{warning}
Never write a bare `except:` that silently swallows every error. You will hide real bugs. Catch the specific exception you expect.
```

## Raising meaningful errors

Your own functions should refuse impossible chemistry.

```{code-cell} ipython3
:tags: [raises-exception]
def kelvin(T):
    if T < 0:
        raise ValueError(f"{T} K is below absolute zero")
    return T

def ideal_gas_pressure(n, V, T):
    if V <= 0:
        raise ValueError("volume must be positive")
    return n * 8.314 * kelvin(T) / V

print(ideal_gas_pressure(1, 0.0224, 273.15))
ideal_gas_pressure(1, 0.0224, -5)
```

You can define your own exception types for a project:

```{code-cell} ipython3
class UnbalancedEquationError(Exception):
    """Raised when atom counts differ between reactants and products."""

def check_balance(reactants, products):
    from collections import Counter
    left, right = Counter(), Counter()
    for counts, coeff in reactants:
        for el, n in counts.items():
            left[el] += n * coeff
    for counts, coeff in products:
        for el, n in counts.items():
            right[el] += n * coeff
    if left != right:
        raise UnbalancedEquationError(f"{dict(left)} != {dict(right)}")
    return True

H2, O2, H2O = {"H": 2}, {"O": 2}, {"H": 2, "O": 1}
print(check_balance([(H2, 2), (O2, 1)], [(H2O, 2)]))
try:
    check_balance([(H2, 1), (O2, 1)], [(H2O, 1)])
except UnbalancedEquationError as e:
    print("Not balanced:", e)
```

## `assert` and simple tests

An `assert` checks something you believe must be true. Collections of asserts become **tests**.

```{code-cell} ipython3
def molar_mass_water():
    return 2 * 1.008 + 15.999

assert math.isclose(molar_mass_water(), 18.015, abs_tol=1e-3)
assert kelvin(0) == 0
print("all tests passed")
```

```{tip}
Professional projects keep tests in files named `test_*.py` and run them with **pytest**. Every time you change your code, `pytest` reruns them all in seconds.
```

## Warnings

Some problems are not fatal but deserve attention.

```{code-cell} ipython3
import warnings
def beer_lambert_conc(A, epsilon, l=1.0):
    if A > 1.5:
        warnings.warn(f"A = {A} is outside the linear range of most spectrometers")
    return A / (epsilon * l)

print(beer_lambert_conc(2.1, 15000))
```

## Debugging strategies

1. **Read the error message**, bottom up.
2. **Print** intermediate values, or use f-strings like `print(f"{n=}, {V=}")`.
3. **Shrink the problem**: reproduce it with the smallest possible input.
4. Use the **debugger**: in Jupyter, run `%debug` in a new cell right after an error to inspect variables at the crash.
5. **Explain the code out loud** ("rubber-duck debugging").

```{code-cell} ipython3
n, V, T = 2.0, 0.050, 300
print(f"{n=}, {V=}, {T=}, {n * 8.314 * T / V = :.0f}")
```

## Exercises

**★ 1.** Predict the exception raised by each line, then check: `int("3.5")`, `[1, 2][5]`, `"Na" + 1`, `{}["Cl"]`.

**★★ 2.** Make `molar_mass` from Chapter 6 raise a clear `ValueError` naming any unknown element symbol.

**★★★ 3.** Write three `assert` tests for `count_atoms` from Chapter 6, including one formula with brackets, and one that should raise an error.
