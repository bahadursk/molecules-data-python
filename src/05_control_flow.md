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

Programs become powerful when they can **decide** and **repeat**. This chapter covers `if`, `for` and `while`.

## Making decisions with `if`

Python uses **indentation** (four spaces) to group the lines that belong to a block.

```{code-cell} ipython3
pH = 8.3

if pH < 7:
    print("acidic")
elif pH > 7:
    print("basic")
else:
    print("neutral")
```

A slightly richer classifier using an indicator:

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

## Repeating with `for`

A `for` loop runs once for each item in a collection.

```{code-cell} ipython3
noble_gases = ["He", "Ne", "Ar", "Kr", "Xe", "Rn"]
for gas in noble_gases:
    print(gas, end=" ")
```

### `range`, `enumerate` and `zip`

```{code-cell} ipython3
for n in range(1, 6):                  # principal quantum numbers 1..5
    print(f"n = {n}: up to {2 * n**2} electrons")
```

```{code-cell} ipython3
for period, gas in enumerate(noble_gases, start=1):
    print(f"period {period}: {gas}")
```

```{code-cell} ipython3
volumes = [0.0, 5.0, 10.0, 15.0, 20.0]       # mL NaOH
pHs = [2.9, 4.1, 4.8, 5.4, 8.7]
for v, p in zip(volumes, pHs):
    print(f"{v:5.1f} mL  pH {p}")
```

### Accumulating results

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

A `while` loop runs as long as a condition is true. It is ideal when you do not know how many steps you need.

**Radioactive decay:** how many half-lives until less than 1 % of a sample of ¹⁴C remains?

```{code-cell} ipython3
fraction = 1.0
half_lives = 0
while fraction >= 0.01:
    fraction /= 2
    half_lives += 1
print(f"{half_lives} half-lives = {half_lives * 5730} years, {fraction:.4%} left")
```

## `break`, `continue` and `else`

```{code-cell} ipython3
# Find the first volume where the pH jumps past 7: the equivalence region
for v, p in zip(volumes, pHs):
    if p > 7:
        print("equivalence near", v, "mL")
        break
else:
    print("never reached equivalence")   # runs only if no break happened
```

```{code-cell} ipython3
masses = [1.20, -0.01, 1.18, None, 1.22]   # some bad readings
good = []
for m in masses:
    if m is None or m < 0:
        continue                            # skip bad values
    good.append(m)
print(good)
```

## Worked example: simulating a titration

Titrating 25.0 mL of 0.100 M HCl with 0.100 M NaOH, one mL at a time.

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

## Exercises

**★ 1.** Loop over `range(1, 21)` and print each atomic number with "even" or "odd".

**★★ 2.** Use a `while` loop to find how many half-lives it takes for ¹³¹I (t½ = 8.02 days) to fall below 0.1 % activity. Give the answer in days.

```{admonition} Solution
:class: dropdown
10 half-lives, 80.2 days.
```

**★★★ 3.** Modify the titration to a **weak** acid (acetic acid, pKa 4.76). Before equivalence use the Henderson–Hasselbalch equation $\mathrm{pH} = \mathrm{p}K_a + \log([A^-]/[HA])$.
