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

Every programmer, however experienced, spends a large part of their time fixing errors. The skill to learn is not avoiding errors but **reading** them quickly and fixing them calmly. When something goes wrong while Python is running, it stops and creates an **exception**, an object describing what happened, and prints a report called a *traceback*. Exceptions are Python's way of refusing to continue with a calculation that no longer makes sense, which is far safer than carrying on and printing a wrong number. This chapter teaches you to understand a traceback, to handle expected problems such as a missing reading without crashing, to make your own functions refuse impossible chemistry, and to find bugs systematically.

## Reading a traceback

A **traceback** is the report Python prints when an exception is not handled. It lists the chain of calls that led to the problem, ending with the line that failed, and its very last line names the type of exception and gives a short message. The cell below contains a classic typo, the digit zero instead of the letter O:

```{code-cell} ipython3
:tags: [raises-exception]
masses = {"H": 1.008, "O": 15.999}
M = 2 * masses["H"] + masses["0"]      # a zero instead of the letter O
```

Read a traceback **from the bottom up**. The last line gives the exception type (`KeyError`) and the problem (`'0'`, the key that was not found). The lines above show where it happened, with an arrow pointing at the line. Inside your own functions the traceback is longer, because it shows each call in turn, but the bottom lines are still the place to start; the earlier lines tell you how the program got there.

## The errors you will meet most

Every exception has a type, and the type already tells you most of what went wrong. A `SyntaxError` is reported before anything runs, because Python cannot understand the code at all. The others happen while the code runs, when a particular value turns out to be unusable.

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

Here are two of them. A `TypeError` happens when an operation does not make sense for the types involved, such as adding text to a number:

```{code-cell} ipython3
:tags: [raises-exception]
"pH " + 7
```

A `ValueError` happens when the type is right but the value is impossible. The logarithm of zero is undefined, so a concentration of exactly zero cannot have a pH:

```{code-cell} ipython3
:tags: [raises-exception]
import math
pH = -math.log10(0.0)
```

## Handling exceptions with `try` / `except`

Sometimes an error is not a bug but an expected situation: an instrument wrote "n/a" for a failed reading, or a file in a folder is not in the right format. **Handling** an exception means catching it and deciding what to do instead of crashing. You put the risky code in a `try` block and the response in an `except` block naming the exception type. If the `try` block succeeds, the `except` block is skipped; if the named exception occurs, Python jumps straight to the `except` block and then carries on after it.

The smallest example converts text that may not be a number:

```{code-cell} ipython3
text = "n/a"
try:
    value = float(text)
except ValueError:
    value = None
print(value)
```

In a loop, this cleans a whole column of raw readings, keeping the good values and reporting the bad ones. `{r!r}` in the f-string prints the value with its quotes, so an empty string is visible as `''`.

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

The full form has four parts. `else` runs only if the `try` block raised no exception, which keeps the "success" code separate from the risky line. `finally` runs no matter what, even after a `return`, which makes it the place to release resources such as an instrument connection.

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

Your own functions should refuse impossible chemistry rather than return nonsense. A temperature below absolute zero or a negative volume is not a small inaccuracy; it means something upstream went wrong, and the earlier you stop, the easier the cause is to find. The `raise` statement creates an exception on purpose. Choose the type that fits (`ValueError` for an impossible value, `TypeError` for the wrong kind of input) and write a message that says what was wrong and what was expected.

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

The traceback now shows two levels: `ideal_gas_pressure` called `kelvin`, which raised the error. The message tells you immediately that a Celsius value was probably passed where kelvin was expected.

You can also define your own exception types for a project. A custom exception is a class (Chapter 10) that inherits from `Exception`; the body can be just a docstring. Its value is that code calling your function can catch *exactly* this problem without accidentally catching unrelated ones. Here a balance checker counts atoms on both sides of an equation with `Counter` (Chapter 7) and raises `UnbalancedEquationError` if they differ:

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

`except ... as e` gives the caught exception a name, so you can print its message.

## `assert` and simple tests

An `assert` statement checks something you believe must be true at that point in the program. If the condition is true, nothing happens; if it is false, Python raises an `AssertionError`. Asserts document your assumptions and catch mistakes the moment they appear.

```{code-cell} ipython3
n_electrons = 10
assert n_electrons % 2 == 0, "expected a closed-shell molecule"
print("check passed")
```

A collection of asserts that compare a function's output with known answers is a **test**. Tests are how you know your code is still correct after you change it. Good test cases are values you can verify independently, such as the molar mass of water, and edge cases such as absolute zero. Remember Chapter 2: compare floats with `math.isclose`, not `==`.

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

Some problems are not fatal but deserve attention: a result that is technically valid but probably unreliable. A **warning** prints a message and lets the program continue, so the calculation still finishes but you are told to be careful. The `warnings` module provides `warnings.warn(message)`. A good example is the Beer–Lambert law, $A = \varepsilon l c$, which holds only up to an absorbance of roughly 1–1.5 on most spectrometers; beyond that, stray light makes the reading too low.

```{code-cell} ipython3
import warnings
def beer_lambert_conc(A, epsilon, l=1.0):
    if A > 1.5:
        warnings.warn(f"A = {A} is outside the linear range of most spectrometers")
    return A / (epsilon * l)

print(beer_lambert_conc(0.6, 15000))
print(beer_lambert_conc(2.1, 15000))
```

## Debugging strategies

A **bug** is code that runs but does the wrong thing, or fails in a way you did not expect. Debugging is detective work, and a methodical approach beats staring at the code:

1. **Read the error message**, bottom up.
2. **Print** intermediate values, or use f-strings like `print(f"{n=}, {V=}")`.
3. **Shrink the problem**: reproduce it with the smallest possible input.
4. Use the **debugger**: in Jupyter, run `%debug` in a new cell right after an error to inspect variables at the crash.
5. **Explain the code out loud** ("rubber-duck debugging").

The `=` inside an f-string prints both the expression and its value, which makes quick checks self-labelling:

```{code-cell} ipython3
n, V, T = 2.0, 0.050, 300
print(f"{n=}, {V=}, {T=}, {n * 8.314 * T / V = :.0f}")
```

A frequent chemistry bug is a unit mismatch that produces no error at all. Printing the intermediate values reveals it: a volume entered in mL instead of L gives a pressure a thousand times too small.

```{code-cell} ipython3
V_mL = 50.0
P_wrong = n * 8.314 * T / V_mL            # V should be in m³
P_right = n * 8.314 * T / (V_mL * 1e-6)
print(f"{P_wrong = :.1f} Pa   {P_right = :.3e} Pa")
```

Chapter 18 shows how the `pint` library catches such unit errors automatically.

```{admonition} Summary
:class: tip
- Read tracebacks from the bottom: the last line names the exception and the problem.
- The exception type (`KeyError`, `ValueError`, `TypeError`, …) points to the cause.
- Use `try` / `except SomeError:` for expected problems; never use a bare `except:`.
- `raise ValueError("message")` makes your functions refuse impossible inputs; custom exceptions inherit from `Exception`.
- `assert` checks assumptions; groups of asserts are tests, run professionally with pytest.
- `warnings.warn` flags doubtful but valid results; `print(f"{x=}")` and `%debug` help find bugs.
```

## Exercises

**★ 1.** Predict the exception raised by each line, then check: `int("3.5")`, `[1, 2][5]`, `"Na" + 1`, `{}["Cl"]`.

```{admonition} Solution
:class: dropdown
`ValueError`, `IndexError`, `TypeError`, `KeyError`.
```

**★★ 2.** Make `molar_mass` from Chapter 6 raise a clear `ValueError` naming any unknown element symbol.

**★★★ 3.** Write three `assert` tests for `count_atoms` from Chapter 6, including one formula with brackets, and one that should raise an error.
