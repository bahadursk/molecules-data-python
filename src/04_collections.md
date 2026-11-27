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

One number is a measurement. Many numbers are **data**. A titration gives a series of burette readings, a spectrum gives hundreds of absorbances, and a periodic table links each symbol to a mass. Storing each value in its own variable (`reading1`, `reading2`, …) quickly becomes impossible, so Python provides **containers**: objects that hold a collection of values under a single name. Choosing the right container is the first design decision in almost every program, because each one is good at a different job. This chapter introduces the four built-in containers, explains what makes each one different, and shows the everyday operations you will use on them for the rest of the book.

| Type | Syntax | Ordered | Changeable | Duplicates | Chemistry use |
|---|---|---|---|---|---|
| list | `[1, 2, 3]` | ✔ | ✔ | ✔ | a series of absorbance readings |
| tuple | `(1, 2, 3)` | ✔ | ✘ | ✔ | x, y, z coordinates of an atom |
| set | `{1, 2, 3}` | ✘ | ✔ | ✘ | the distinct ions in a sample |
| dict | `{"H": 1.008}` | ✔ | ✔ | keys ✘ | a lookup table of atomic masses |

"Ordered" means the container remembers the order in which you put things in, so you can ask for the first or the third item. "Changeable" (the technical word is *mutable*) means you can add, remove or replace items after the container has been created. "Duplicates" says whether the same value may appear twice.

## Lists

A **list** is an ordered, changeable sequence of values, written as items separated by commas inside square brackets. It is the general-purpose container and the one you will reach for most often, whenever you have a series of things in a meaningful order: readings in the order they were taken, elements in a group, or compounds in a screening queue. A list can hold any kind of value, and the items need not all be the same type, although in practice they usually are.

The simplest list holds a few values, and `len()` tells you how many:

```{code-cell} ipython3
noble_gases = ["He", "Ne", "Ar"]
print(noble_gases)
print(len(noble_gases))
```

Lists are indexed and sliced exactly like strings (Chapter 3): positions start at 0, negative positions count from the end, and `[start:stop]` takes a range. The method `append()` adds one item at the end. Built-in functions such as `max()`, `min()`, `sum()` and `sorted()` work on any list of numbers, so many simple summaries need no extra code at all.

```{code-cell} ipython3
halogens = ["F", "Cl", "Br", "I"]
halogens.append("At")
print(halogens, len(halogens))
print(halogens[0], halogens[-1], halogens[1:3])

boiling_points = [-188.1, -34.0, 58.8, 184.3, 337.0]   # °C
print("highest bp:", max(boiling_points))
print("sorted:", sorted(boiling_points, reverse=True))
```

The two lists above are *parallel*: the third boiling point belongs to the third halogen. Section 4.5 shows a safer way to keep related values together.

Because lists are mutable, you can change them in place. Assigning to an index replaces one item, `insert(position, value)` adds an item anywhere, and `remove(value)` or `pop(position)` take items out. This is how you correct a mistyped reading or add a blank measurement without rebuilding the whole list.

```{code-cell} ipython3
readings = [0.112, 0.224, 0.338, 0.995, 0.561]   # absorbance; one obvious outlier
readings[3] = 0.449        # correct a typo
readings.insert(0, 0.000)  # add the blank
print(readings)
print("mean:", sum(readings) / len(readings))
```

`sort()` and `sorted()` look alike but behave differently. The method `readings.sort()` reorders the list itself and returns nothing, while the function `sorted(readings)` leaves the original alone and returns a new sorted list. Use `sorted()` when you still need the original order.

```{code-cell} ipython3
masses = [35.45, 1.008, 12.011]
print(sorted(masses), masses)   # original unchanged
masses.sort()
print(masses)                   # now changed in place
```

```{warning}
`b = a` does **not** copy a list: both names point to the same object. Use `b = a.copy()` or `b = list(a)`.
```

This catches everyone once. A variable name is only a label attached to an object, so `b = a` attaches a second label to the *same* list. Changing the list through either name changes it for both:

```{code-cell} ipython3
a = [1, 2, 3]
b = a
b.append(4)
print(a)   # a changed too!

c = a.copy()
c.append(5)
print(a, c)   # a is safe this time
```

## Tuples

A **tuple** is an ordered sequence like a list, but it is *immutable*: once made, it cannot be changed. You write it with parentheses instead of square brackets. The restriction is useful. A tuple signals to anyone reading your code that the values belong together and are not meant to change, such as the x, y and z coordinates of one atom, a (temperature, pressure) pair, or the constants of a fitted equation. Python will refuse an accidental change rather than let it silently corrupt your data.

```{code-cell} ipython3
point = (0.0, 0.757, -0.469)     # x, y, z in Å
print(point[1])
try:
    point[1] = 0.8
except TypeError as err:
    print("TypeError:", err)
```

The most convenient feature of tuples is **unpacking**: assigning a tuple to several names at once gives each name one item. The same happens automatically in a `for` loop over a list of tuples, which makes coordinate tables pleasant to read.

```{code-cell} ipython3
x, y, z = point
print(x, y, z)

water = (("O", 0.000, 0.000, 0.117),
         ("H", 0.000, 0.757, -0.469),
         ("H", 0.000, -0.757, -0.469))   # Å

for symbol, x, y, z in water:          # tuple unpacking
    print(f"{symbol} {x:7.3f} {y:7.3f} {z:7.3f}")
```

Unpacking also lets a function return several values at once (Chapter 6): `mean, std = summarise(data)`.

## Sets

A **set** is an unordered collection in which every item appears at most once. You write it with curly braces, or create it from another collection with `set()`, which silently drops duplicates. Sets answer membership questions very fast and support the operations of set theory from mathematics: the intersection `&` (items in both), the union `|` (items in either) and the difference `-` (items in the first but not the second). Whenever the question is "which distinct things are there?" or "what do these two groups share?", a set is the natural tool.

Removing duplicates is the most basic use:

```{code-cell} ipython3
elements_seen = ["C", "H", "H", "O", "C", "H"]
print(set(elements_seen))
print(len(set(elements_seen)), "distinct elements")
```

Comparing the ions found in two water samples is a direct application of the set operations:

```{code-cell} ipython3
sample_A = {"Na+", "Cl-", "K+", "SO4^2-"}
sample_B = {"Na+", "Cl-", "Ca2+", "NO3-"}
print("in both:", sample_A & sample_B)
print("in either:", sample_A | sample_B)
print("only in A:", sample_A - sample_B)
print("unique elements in glucose formula:", set("CHO"))
```

Because a set has no order, it cannot be indexed: `sample_A[0]` is an error. The order in which a set prints may also vary, so convert it with `sorted()` if you need a predictable listing.

## Dictionaries: a mini periodic table

A **dictionary** (type `dict`) stores pairs: each **key** is linked to a **value**, and you look up the value by giving the key in square brackets. It works like a real dictionary, where a word leads to its definition, or like a periodic table, where a symbol leads to a mass. Keys must be unique and unchangeable (strings and numbers are typical), while values can be anything at all, including lists or other dictionaries. Dictionaries are the right choice whenever you want to find information by a *name* rather than by a position.

The basic operations are creating, looking up and adding:

```{code-cell} ipython3
density = {"water": 0.997, "ethanol": 0.789}   # g/mL
print(density["ethanol"])
density["hexane"] = 0.655       # add a new key
print(density)
```

Looking up a key that is not there raises a `KeyError`. Test first with `in`, or use `get(key, default)`, which returns the default instead of failing. This matters when your data might be incomplete.

```{code-cell} ipython3
atomic_mass = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999,
               "Na": 22.990, "S": 32.06, "Cl": 35.45}
print(atomic_mass["Cl"])
atomic_mass["Fe"] = 55.845          # add an entry
print("Fe" in atomic_mass, "U" in atomic_mass)
print(atomic_mass.get("U", "unknown"))
```

The methods `keys()`, `values()` and `items()` let you walk through a dictionary. `items()` gives (key, value) tuples, which unpack neatly in a loop. Combining a dictionary of masses with a dictionary of atom counts gives a molar-mass calculator in four lines:

```{code-cell} ipython3
caffeine = {"C": 8, "H": 10, "N": 4, "O": 2}
M = 0
for element, count in caffeine.items():
    M += atomic_mass[element] * count
print(f"M(caffeine) = {M:.2f} g/mol")
```

## Nested data

Containers can hold other containers, and real data is usually nested this way. A list of dictionaries, one dictionary per row with the same keys in each, is the shape of a table: each dictionary is a compound and each key is a column. It keeps related values together, so the boiling point of toluene can never drift out of step with its name as it might with parallel lists. It is also exactly the shape that pandas will turn into a table in Chapter 15.

Reaching into nested data uses one pair of brackets per level:

```{code-cell} ipython3
solvents = [
    {"name": "water",    "bp": 100.0, "density": 0.997, "polar": True},
    {"name": "ethanol",  "bp": 78.4,  "density": 0.789, "polar": True},
    {"name": "hexane",   "bp": 68.7,  "density": 0.655, "polar": False},
    {"name": "toluene",  "bp": 110.6, "density": 0.867, "polar": False},
    {"name": "DMSO",     "bp": 189.0, "density": 1.100, "polar": True},
]
print(solvents[1])            # the second row
print(solvents[1]["bp"])      # one value from it
```

To sort the rows by a column, `sorted` takes a `key` argument: a small function that tells it what to sort by. Here `lambda s: s["bp"]` is a one-line anonymous function (Chapter 6) meaning "given a row `s`, use its boiling point".

```{code-cell} ipython3
for s in sorted(solvents, key=lambda s: s["bp"]):
    tag = "polar" if s["polar"] else "non-polar"
    print(f"{s['name']:8s} bp {s['bp']:6.1f} °C  ρ {s['density']:.3f} g/mL  {tag}")
```

A quick colourful look at the same data shows why plotting is worth learning early. The lists `names`, `bps` and `colors` are built with list comprehensions, the compact loops of Chapter 7:

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

```{admonition} Summary
:class: tip
- A **list** `[...]` is an ordered, changeable sequence: use it for series of values.
- A **tuple** `(...)` is an ordered, unchangeable sequence: use it for fixed groups such as coordinates, and unpack it with `x, y, z = t`.
- A **set** `{...}` holds distinct items with no order: use it to remove duplicates and compare groups with `&`, `|`, `-`.
- A **dict** `{key: value}` looks values up by name: use `d[key]`, `key in d`, `d.get(key, default)` and `d.items()`.
- `b = a` shares a list; `a.copy()` makes an independent one.
- A list of dictionaries is a table, the bridge to pandas.
```

## Exercises

**★ 1.** Make a list of the first ten elements' symbols and print the 3rd to 6th.

```{admonition} Solution
:class: dropdown
`symbols = ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne"]` then `symbols[2:6]` gives `['Li', 'Be', 'B', 'C']`.
```

**★★ 2.** Given `{"Na": 1, "Cl": 1}` and `{"Mg": 1, "Cl": 2}`, compute the molar masses of NaCl and MgCl₂ (Mg 24.305).

```{admonition} Solution
:class: dropdown
Add `atomic_mass["Mg"] = 24.305`, then reuse the loop above: 58.44 and 95.21 g/mol.
```

**★★★ 3.** Store the electron configuration of the first 18 elements as a dictionary mapping symbol to a list of shell occupancies, e.g. `"Na": [2, 8, 1]`. Use it to find every element with one valence electron.
