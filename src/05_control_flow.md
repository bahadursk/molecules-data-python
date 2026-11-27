---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Decisions and Loops

```{admonition} 🟢 Beginner · school & first-year
:class: level-1
Builds on Chapters 1–4.
```

So far every program has run from top to bottom, executing each line exactly once. Real tasks need more. A program that classifies a solution must do one thing if the pH is low and another if it is high, and a program that processes a spectrum must repeat the same calculation for every data point. These two abilities, **deciding** and **repeating**, are called *control flow* because they control the order in which lines run. With them, a dozen lines of code can process a million measurements, and the computer finally saves you work rather than just doing arithmetic. This chapter covers the `if` statement for decisions and the `for` and `while` loops for repetition, then puts them together to simulate a titration curve.

## Making decisions with `if`

An `if` statement runs a block of code only when a condition is true. The condition is any expression that gives a Boolean (Chapter 2), such as `pH < 7`. The line ends with a colon, and the lines that belong to the `if` are **indented** by four spaces. Indentation is not decoration in Python; it is how the language knows which lines are inside the block and which come after it. Most languages use braces for this, but Python makes the visual layout and the meaning the same thing, which keeps code readable.

The simplest form has a single branch:

```{code-cell} ipython3
temperature = 105    # °C
if temperature > 100:
    print("water would boil")
print("this line always runs")
```

`else` adds a branch for when the condition is false, and `elif` (short for "else if") adds further conditions, which are tested in order. Python runs the block of the *first* condition that is true and skips all the rest, so only one branch ever runs.

```{code-cell} ipython3
pH = 8.3

if pH < 7:
    print("acidic")
elif pH > 7:
    print("basic")
else:
    print("neutral")
```

Because the tests are checked in order, the order matters. Here the boundaries of each colour range follow from the earlier tests: by the time Python reaches `elif pH <= 10.0`, it already knows the pH is at least 8.2. The code is wrapped in a function so it can be reused for any pH; `def` and `return` are explained fully in Chapter 6.

```{code-cell} ipython3
def indicator_colour(pH):
    """Colour of phenolphthalein at a given pH."""
    if pH < 8.2:
        return "colourless"
    elif pH <= 10.0:
        return "pink"
    else:
        return "deep magenta"

for p in (3.0, 8.5, 12.0):
    print(p, indicator_colour(p))
```

For a simple two-way choice there is also a one-line form, the *conditional expression* `value_if_true if condition else value_if_false`. It is handy for labels:

```{code-cell} ipython3
n_electrons = 15
print("radical" if n_electrons % 2 else "closed shell")
```

## Repeating with `for`

A `for` loop runs a block once for each item in a collection, giving the item a name each time round. You use it whenever you have a known collection to walk through: every element in a list, every line of a file, every row of a table. The loop variable (`gas` below) takes the value of the first item, the indented block runs, then the variable takes the next item, and so on until the collection is exhausted.

```{code-cell} ipython3
noble_gases = ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]
for gas in noble_gases:
    print(gas, end=" ")
```

`end=" "` tells `print` to finish with a space instead of a new line, so the output stays on one row. A loop works on any sequence, including a string, where it visits each character:

```{code-cell} ipython3
for ch in "H2O":
    print(ch, ch.isdigit())
```

### `range`, `enumerate` and `zip`

Three built-in helpers make loops more useful. `range(start, stop)` produces whole numbers from `start` up to but not including `stop`, which is how you loop a fixed number of times or over a sequence of integers such as quantum numbers. With one argument, `range(5)` counts 0, 1, 2, 3, 4.

```{code-cell} ipython3
for n in range(1, 6):                  # principal quantum numbers 1..5
    print(f"n = {n}: up to {2 * n**2} electrons")
```

`enumerate(collection)` gives each item *together with* its position, so you never need a separate counter variable. The argument `start=1` makes counting begin at 1, which suits periods and sample numbers.

```{code-cell} ipython3
for period, gas in enumerate(noble_gases, start=1):
    print(f"period {period}: {gas}")
```

`zip(a, b)` walks through two (or more) collections side by side, pairing their items. It is the natural way to process paired data such as volumes and pH readings from a titration.

```{code-cell} ipython3
volumes = [0.0, 5.0, 10.0, 15.0, 20.0]       # mL NaOH
pHs = [2.9, 4.1, 4.8, 5.4, 8.7]
for v, p in zip(volumes, pHs):
    print(f"{v:5.1f} mL  pH {p}")
```

### Accumulating results

A very common pattern is to start with an empty result before the loop and add to it on every pass. For a running total you start with 0 and add; for a collection of results you start with an empty list and `append`. This *accumulator* pattern is the basis of sums, averages and tables of computed values.

```{code-cell} ipython3
masses = [12.011, 1.008, 1.008, 15.999]
total = 0
for m in masses:
    total += m          # same as total = total + m
print(f"total {total:.3f}")
```

The Bohr model gives the energy levels of the hydrogen atom as $E_n = -13.6\ \mathrm{eV}/n^2$. Accumulating them in a list and then drawing them shows why the visible Balmer lines all end on $n = 2$:

```{code-cell} ipython3
# Energy levels of hydrogen, E_n = -13.6 eV / n^2
levels = []
for n in range(1, 7):
    levels.append(-13.6 / n**2)
print([round(E, 3) for E in levels])

import matplotlib.pyplot as plt
for n, E in enumerate(levels, start=1):
    plt.hlines(E, 0, 1, colors=plt.cm.plasma(n / 7), linewidth=3)
    plt.text(1.02, E, f"n={n}", va="center")
for n_upper, colour in zip(range(3, 7), ["red", "cyan", "blue", "violet"]):
    x = 0.15 * (n_upper - 2)
    plt.annotate("", xy=(x, levels[1]), xytext=(x, levels[n_upper - 1]),
                 arrowprops=dict(arrowstyle="->", color=colour, lw=2))
plt.ylabel("Energy / eV")
plt.title("Hydrogen energy levels and the Balmer series")
plt.xticks([])
plt.xlim(0, 1.2)
plt.show()
```

## Repeating with `while`

A `while` loop repeats its block as long as a condition stays true, checking the condition before every pass. It is the right loop when you do not know in advance how many steps you will need, only when to stop: keep heating until the solid melts, keep iterating until the answer stops changing, keep halving until the sample is nearly gone. Something inside the loop must eventually make the condition false, otherwise the loop runs for ever; if that happens in a notebook, use the stop button to interrupt the kernel.

A minimal example doubles a number until it passes a limit:

```{code-cell} ipython3
cells = 1
doublings = 0
while cells < 1000:
    cells *= 2
    doublings += 1
print(doublings, "doublings give", cells, "cells")
```

**Radioactive decay:** how many half-lives until less than 1 % of a sample of ¹⁴C remains? Each half-life halves the remaining fraction, and we stop as soon as it falls below 0.01.

```{code-cell} ipython3
fraction = 1.0
half_lives = 0
while fraction >= 0.01:
    fraction /= 2
    half_lives += 1
print(f"{half_lives} half-lives = {half_lives * 5730} years, {fraction:.4%} left")
```

The format `.4%` multiplies by 100 and adds a percent sign, a convenient way to print fractions.

## `break`, `continue` and `else`

Two keywords change a loop's normal rhythm. `break` leaves the loop immediately, which is useful when you are searching and have found what you wanted. `continue` skips the rest of the current pass and jumps straight to the next item, which is useful for ignoring bad data. A loop can also have an `else` block, which runs only if the loop finished without hitting `break`; it answers the question "did the search fail?" without an extra flag variable.

```{code-cell} ipython3
# Find the first volume where the pH jumps past 7: the equivalence region
for v, p in zip(volumes, pHs):
    if p > 7:
        print("equivalence near", v, "mL")
        break
else:
    print("never reached equivalence")   # runs only if no break happened
```

Here `continue` filters out missing and impossible readings. `None` is Python's value for "nothing here", often used for a missing measurement, and `is None` is the correct way to test for it.

```{code-cell} ipython3
masses = [1.20, -0.01, 1.18, None, 1.22]   # some bad readings
good = []
for m in masses:
    if m is None or m < 0:
        continue                            # skip bad values
    good.append(m)
print(good)
```

Note the order inside the `if`: `m is None` is tested first, and because `or` stops as soon as it finds a true part, `m < 0` is never evaluated for `None`, which would otherwise raise an error.

## Worked example: simulating a titration

Decisions and loops together can simulate a whole experiment. Consider titrating 25.0 mL of 0.100 M HCl with 0.100 M NaOH, one mL at a time. At each step we compute the moles of acid and base, and then the chemistry branches into three cases: before equivalence the excess H⁺ sets the pH, after equivalence the excess OH⁻ sets the pOH, and exactly at equivalence a strong acid–strong base mixture is neutral. Volumes are divided by 1000 to convert mL to L, because concentrations are in mol L⁻¹.

```{code-cell} ipython3
import math
Va, Ca, Cb = 25.0, 0.100, 0.100
n_acid = Va * Ca / 1000

vols, ph_values = [], []
for i in range(0, 51):
    Vb = i * 1.0
    n_base = Vb * Cb / 1000
    V_total = (Va + Vb) / 1000
    if n_base < n_acid:
        pH = -math.log10((n_acid - n_base) / V_total)
    elif n_base > n_acid:
        pOH = -math.log10((n_base - n_acid) / V_total)
        pH = 14 - pOH
    else:
        pH = 7.0
    vols.append(Vb)
    ph_values.append(pH)

plt.scatter(vols, ph_values, c=ph_values, cmap="rainbow", s=30)
plt.colorbar(label="pH")
plt.axvline(25, ls="--", color="grey")
plt.xlabel("Volume of NaOH / mL")
plt.ylabel("pH")
plt.title("Strong acid / strong base titration")
plt.show()
```

The steep jump around 25 mL is the equivalence point, and the simulation reproduces it from nothing more than stoichiometry, a loop and an `if`.

```{admonition} Summary
:class: tip
- `if` / `elif` / `else` runs the first block whose condition is true; indentation marks the block.
- `for item in collection:` repeats once per item; `range`, `enumerate` and `zip` supply numbers, positions and pairs.
- The accumulator pattern starts with `0` or `[]` before the loop and adds on each pass.
- `while condition:` repeats until the condition is false; make sure something changes it.
- `break` exits a loop, `continue` skips to the next pass, and a loop's `else` runs only if no `break` happened.
```

## Exercises

**★ 1.** Loop over `range(1, 21)` and print each atomic number with "even" or "odd".

```{admonition} Solution
:class: dropdown
`for Z in range(1, 21): print(Z, "even" if Z % 2 == 0 else "odd")`
```

**★★ 2.** Use a `while` loop to find how many half-lives it takes for ¹³¹I (t½ = 8.02 days) to fall below 0.1 % activity. Give the answer in days.

```{admonition} Solution
:class: dropdown
10 half-lives, 80.2 days.
```

**★★★ 3.** Modify the titration to a **weak** acid (acetic acid, pKa 4.76). Before equivalence use the Henderson–Hasselbalch equation $\mathrm{pH} = \mathrm{p}K_a + \log([A^-]/[HA])$.
