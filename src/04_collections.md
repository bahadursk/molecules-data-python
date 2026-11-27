---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Lists, Tuples, Sets and Dictionaries

```{admonition} 🟢 Beginner · school & first-year
:class: level-1
Builds on Chapters 1–3.
```

One number is a measurement. Many numbers are **data**. Python has four built-in containers for collections of values.

| Type | Syntax | Ordered | Changeable | Duplicates | Chemistry use |
|---|---|---|---|---|---|
| list | `[1, 2, 3]` | ✔ | ✔ | ✔ | a series of absorbance readings |
| tuple | `(1, 2, 3)` | ✔ | ✘ | ✔ | x, y, z coordinates of an atom |
| set | `{1, 2, 3}` | ✘ | ✔ | ✘ | the distinct ions in a sample |
| dict | `{"H": 1.008}` | ✔ | ✔ | keys ✘ | a lookup table of atomic masses |

## Lists

```{code-cell} ipython3
halogens = ["F", "Cl", "Br", "I"]
halogens.append("At")
print(halogens, len(halogens))
print(halogens[0], halogens[-1], halogens[1:3])

boiling_points = [-188.1, -34.0, 58.8, 184.3, 337.0]   # °C
print("highest bp:", max(boiling_points))
print("sorted:", sorted(boiling_points, reverse=True))
```

Lists can be changed in place:

```{code-cell} ipython3
readings = [0.112, 0.224, 0.338, 0.995, 0.561]   # absorbance; one obvious outlier
readings[3] = 0.449        # correct a typo
readings.insert(0, 0.000)  # add the blank
print(readings)
print("mean:", sum(readings) / len(readings))
```

```{warning}
`b = a` does **not** copy a list: both names point to the same object. Use `b = a.copy()` or `b = list(a)`.
```

```{code-cell} ipython3
a = [1, 2, 3]
b = a
b.append(4)
print(a)   # a changed too!
```

## Tuples

Tuples are lists that cannot change, which makes them ideal for fixed facts.

```{code-cell} ipython3
water = (("O", 0.000, 0.000, 0.117),
         ("H", 0.000, 0.757, -0.469),
         ("H", 0.000, -0.757, -0.469))   # Å

for symbol, x, y, z in water:          # tuple unpacking
    print(f"{symbol} {x:7.3f} {y:7.3f} {z:7.3f}")
```

Unpacking lets a function return several values at once (Chapter 6).

## Sets

```{code-cell} ipython3
sample_A = {"Na+", "Cl-", "K+", "SO4^2-"}
sample_B = {"Na+", "Cl-", "Ca2+", "NO3-"}
print("in both:", sample_A & sample_B)
print("in either:", sample_A | sample_B)
print("only in A:", sample_A - sample_B)
print("unique elements in glucose formula:", set("CHO"))
```

## Dictionaries: a mini periodic table

A dictionary maps **keys** to **values**.

```{code-cell} ipython3
atomic_mass = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
               "Na": 22.990, "S": 32.06, "Cl": 35.45}
print(atomic_mass["Cl"])
atomic_mass["Fe"] = 55.845          # add an entry
print("Fe" in atomic_mass, "U" in atomic_mass)
print(atomic_mass.get("U", "unknown"))
```

Combining a dictionary of masses with a dictionary of atom counts gives a molar-mass calculator:

```{code-cell} ipython3
caffeine = {"C": 8, "H": 10, "N": 4, "O": 2}
M = 0
for element, count in caffeine.items():
    M += atomic_mass[element] * count
print(f"M(caffeine) = {M:.2f} g/mol")
```

## Nested data

Real data is usually nested: a list of dictionaries is the shape of a table, and the shape pandas will read in Chapter 15.

```{code-cell} ipython3
solvents = [
    {"name": "water",    "bp": 100.0, "density": 0.997, "polar": True},
    {"name": "ethanol",  "bp": 78.4,  "density": 0.789, "polar": True},
    {"name": "hexane",   "bp": 68.7,  "density": 0.655, "polar": False},
    {"name": "toluene",  "bp": 110.6, "density": 0.867, "polar": False},
    {"name": "DMSO",     "bp": 189.0, "density": 1.100, "polar": True},
]
for s in sorted(solvents, key=lambda s: s["bp"]):
    tag = "polar" if s["polar"] else "non-polar"
    print(f"{s['name']:8s} bp {s['bp']:6.1f} °C  ρ {s['density']:.3f} g/mL  {tag}")
```

A quick colourful look at the same data:

```{code-cell} ipython3
import matplotlib.pyplot as plt
names = [s["name"] for s in solvents]
bps = [s["bp"] for s in solvents]
colors = ["#0f9d8a" if s["polar"] else "#f2a93b" for s in solvents]
plt.bar(names, bps, color=colors)
plt.ylabel("Boiling point / °C")
plt.title("Polar (teal) and non-polar (amber) solvents")
plt.show()
```

## Exercises

**★ 1.** Make a list of the first ten elements' symbols and print the 3rd to 6th.

**★★ 2.** Given `{"Na": 1, "Cl": 1}` and `{"Mg": 1, "Cl": 2}`, compute the molar masses of NaCl and MgCl₂ (Mg 24.305).

```{admonition} Solution
:class: dropdown
Add `atomic_mass["Mg"] = 24.305`, then reuse the loop above: 58.44 and 95.21 g/mol.
```

**★★★ 3.** Store the electron configuration of the first 18 elements as a dictionary mapping symbol to a list of shell occupancies, e.g. `"Na": [2, 8, 1]`. Use it to find every element with one valence electron.
