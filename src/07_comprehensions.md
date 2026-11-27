---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Pythonic Code: Comprehensions, Generators and Idioms

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 1–6.
```

Code that is called "Pythonic" solves a problem in the way the language was designed to be used: short, clear, and relying on built-in tools rather than long hand-written loops. This is not about elegance for its own sake. Shorter code has fewer places for bugs to hide, code that reads like a sentence is easier to check against the chemistry it implements, and every scientific library you will use (NumPy, pandas, RDKit) is written and documented in this style, so learning it lets you read their examples. This chapter covers the idioms you will meet most: comprehensions for building collections, generators for producing values on demand, unpacking, the `itertools` module for combinations, and a few small helpers from the standard library.

## List comprehensions

A **list comprehension** builds a new list from an existing collection in a single line, by describing what each new item should be. It replaces the three-line accumulator pattern from Chapter 5 (create an empty list, loop, append). The two pieces of code below do the same job; the comprehension states the intent, "the Kelvin value of each temperature", directly.

```{code-cell} ipython3
temps_C = [-78.5, 0, 25, 37, 100]

temps_K = []                        # the loop version
for t in temps_C:
    temps_K.append(t + 273.15)
print(temps_K)

temps_K = [t + 273.15 for t in temps_C]   # the comprehension
print(temps_K)
```

The general form is `[expression for item in iterable if condition]`. The `for` part says what to loop over, the expression at the front says what to put in the new list, and the optional `if` part keeps only the items that pass a test. Reading it aloud from the middle outwards works well: "for each `z` in the atomic numbers, if `z` is a noble gas, keep `z`".

```{code-cell} ipython3
atomic_numbers = range(1, 37)
# Noble gases up to krypton
noble = [z for z in atomic_numbers if z in (2, 10, 18, 36)]
squares = [n**2 for n in range(1, 5)]
print(noble, squares)
```

A comprehension can transform and filter at the same time. Here only the measurable absorbances are kept and converted to transmittance with $T = 10^{-A}$:

```{code-cell} ipython3
absorbance = [0.12, 0.45, 3.80, 0.88, -0.01]
transmittance = [round(10 ** -A, 3) for A in absorbance if 0 <= A <= 2]
print(transmittance)
```

Use a comprehension when the logic fits comfortably on one line. If you find yourself nesting several conditions, a normal loop is clearer.

## Dictionary and set comprehensions

The same idea works for dictionaries and sets, written with curly braces. A **dictionary comprehension** has `key: value` at the front and builds a new mapping, which is the quickest way to convert every value in a lookup table, for example turning pKa values into Ka values. A **set comprehension** has a single expression and collects the distinct results.

```{code-cell} ipython3
import math
pKa = {"acetic": 4.76, "formic": 3.75, "benzoic": 4.20, "phenol": 9.95, "HF": 3.17}
Ka = {acid: 10 ** -p for acid, p in pKa.items()}
strong_ish = {acid for acid, p in pKa.items() if p < 4}
print({k: f"{v:.2e}" for k, v in Ka.items()})
print(strong_ish)
```

## Nested comprehensions: building orbitals

A comprehension can contain more than one `for`, and they run like nested loops, the leftmost being the outer loop. This is how you generate every combination of two quantum numbers. For principal quantum number $n$, the angular momentum quantum number $l$ takes the values $0$ to $n-1$, labelled s, p, d, f:

```{code-cell} ipython3
orbitals = [f"{n}{'spdf'[l]}" for n in range(1, 5) for l in range(n)]
print(orbitals)
```

The expression `'spdf'[l]` indexes a string to turn the number $l$ into its letter. Electrons fill orbitals in the order given by the **Aufbau (n + l) rule**: lower $n + l$ first, and for equal $n + l$, lower $n$ first. Python's `sorted` can sort by a tuple key, comparing the first item and then breaking ties with the second, which is exactly this rule. The resulting order then drives an electron-configuration function:

```{code-cell} ipython3
aufbau = sorted(((n, l) for n in range(1, 8) for l in range(min(n, 4))), key=lambda o: (o[0] + o[1], o[0]))
capacity = [2 * (2 * l + 1) for n, l in aufbau]
print([f"{n}{'spdf'[l]}" for n, l in aufbau[:10]])

def electron_configuration(Z):
    config, left = [], Z
    for (n, l), cap in zip(aufbau, capacity):
        if left <= 0:
            break
        e = min(cap, left)
        config.append(f"{n}{'spdf'[l]}{e}")
        left -= e
    return " ".join(config)

for Z, el in [(6, "C"), (11, "Na"), (26, "Fe"), (35, "Br")]:
    print(f"{el:2s} {electron_configuration(Z)}")
```

The rule gives the textbook configurations for these elements; a few transition metals such as Cr and Cu are known exceptions that the simple rule does not capture.

## Generators: lazy sequences

A list comprehension builds the entire list in memory at once. A **generator** instead produces its values one at a time, only when they are asked for, and then forgets them. This is called *lazy* evaluation. It matters when the sequence is huge (millions of lines in a trajectory file) or even endless, and when you only need to pass the values to a function such as `sum()` that consumes them one by one. A **generator expression** looks like a list comprehension but uses round brackets.

```{code-cell} ipython3
squares_list = [n**2 for n in range(5)]     # built immediately
squares_gen = (n**2 for n in range(5))      # built on demand
print(squares_list)
print(squares_gen)                          # just a recipe, no values yet
print(sum(squares_gen))                     # values produced as sum asks for them
```

When a generator expression is the only argument to a function, you can drop its extra brackets, which reads naturally:

```{code-cell} ipython3
total_mass = sum(m * n for m, n in [(12.011, 6), (1.008, 6)])   # benzene
print(total_mass)
```

A **generator function** is a function that uses `yield` instead of `return`. Each `yield` hands out one value and pauses the function, keeping all its local variables, until the next value is requested. This lets you write an endless stream, here the remaining amount of radioactive cobalt-60 at regular time steps, and let the caller decide when to stop:

```{code-cell} ipython3
def decay(N0, half_life, dt):
    N, t = N0, 0.0
    while True:
        yield t, N
        t += dt
        N *= 0.5 ** (dt / half_life)

for t, N in decay(1000, 5.27, 2.0):   # cobalt-60, years
    if N < 100:
        break
    print(f"t = {t:4.1f} y  N = {N:6.1f}")
```

The `while True` loop would never end on its own, but because the generator only runs when the `for` loop asks for the next value, the `break` stops everything cleanly.

## Unpacking tricks

Chapter 4 introduced unpacking a tuple into separate names. Python extends this in useful ways. A starred name such as `*middle` collects "everything else" into a list, so you can pull out the first and last items of any sequence. Two variables can be swapped in one line, because the right-hand side is packed into a tuple before anything is assigned. And `zip(*rows)` transposes a table: the star spreads the rows out as separate arguments to `zip`, which then pairs up their first items, their second items and so on, turning a list of (x, y, z) points into separate x, y and z sequences.

```{code-cell} ipython3
first, *middle, last = ["Li", "Na", "K", "Rb", "Cs"]
print(first, middle, last)

a, b = 1, 2
a, b = b, a                 # swap without a temporary variable
print(a, b)

coords = [(0.0, 0.0, 0.0), (0.96, 0.0, 0.0)]
xs, ys, zs = zip(*coords)    # transpose
print(xs)
```

## `itertools` for combinatorial chemistry

The standard-library module `itertools` contains fast building blocks for looping over combinations. `combinations(items, k)` gives every unordered group of `k` different items, `permutations(items, k)` gives every ordered arrangement, and `product(a, b, ...)` gives every way of picking one item from each collection (with `repeat=2` meaning "pick twice from the same collection"). These are exactly the operations of combinatorial chemistry, where a library is built by choosing one building block for each position on a scaffold.

The difference between combinations and permutations is easiest to see on a small example:

```{code-cell} ipython3
from itertools import combinations, product, permutations

print(list(combinations("ABC", 2)))    # order does not matter
print(list(permutations("ABC", 2)))    # order matters
```

Applied to chemistry, `combinations` lists the possible interhalogen pairs and `product` enumerates a library of disubstituted benzenes from four substituents:

```{code-cell} ipython3
halides = ["F", "Cl", "Br", "I"]
print("Interhalogen pairs:", ["".join(p) for p in combinations(halides, 2)])

r_groups = ["H", "CH3", "OH", "Cl"]
library = list(product(r_groups, repeat=2))
print(len(library), "disubstituted benzenes (before symmetry):", library[:6], "...")
```

The 16 pairs include duplicates: (CH₃, OH) and (OH, CH₃) are the same *para* compound viewed from opposite ends. RDKit's canonical SMILES (Chapter 19) gives one unique string per molecule, so storing the strings in a set removes the duplicates automatically:

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Draw

smi = {"H": "[H]", "CH3": "C", "OH": "O", "Cl": "Cl"}
mols, legends = [], []
seen = set()
for r1, r2 in library:
    m = Chem.MolFromSmiles(f"{smi[r1]}c1ccc({smi[r2]})cc1")
    can = Chem.MolToSmiles(m)
    if can not in seen:                  # remove duplicates by symmetry
        seen.add(can)
        mols.append(m)
        legends.append(f"{r1} / {r2}")
print(len(mols), "unique para-disubstituted benzenes")
Draw.MolsToGridImage(mols, legends=legends, molsPerRow=5, subImgSize=(160, 140))
```

## Other idioms worth knowing

A handful of small built-in tools appear constantly in scientific code. `all()` returns `True` if every item passes a test and `any()` returns `True` if at least one does; together with a generator expression they turn a quality-control check into one readable line. The conditional expression `a if condition else b` (Chapter 5) picks between two values. From the `collections` module, `Counter` counts how many times each item occurs, and `defaultdict(list)` is a dictionary that creates an empty list the first time you use a new key, which makes grouping items into categories effortless.

```{code-cell} ipython3
# any / all
pHs = [6.8, 7.1, 7.4, 7.0]
print(all(6.5 < p < 7.5 for p in pHs), any(p > 7.3 for p in pHs))

# Conditional expression
T = 400
phase = "gas" if T > 373.15 else "liquid"
print(phase)
```

`Counter` on a SMILES string counts each character, including bond symbols and ring digits, so it is a quick look rather than a true atom count:

```{code-cell} ipython3
from collections import Counter, defaultdict
print(Counter("CC(=O)Oc1ccccc1C(=O)O".upper().replace("(", "").replace(")", "")))
print(Counter(["Na", "Cl", "Na", "K"]).most_common(1))

groups = defaultdict(list)
for el, block in [("Na", "s"), ("Fe", "d"), ("C", "p"), ("K", "s"), ("Cu", "d")]:
    groups[block].append(el)
print(dict(groups))
```

```{admonition} Summary
:class: tip
- `[expr for x in data if test]` builds a list in one line; `{k: v for ...}` and `{x for ...}` build dicts and sets.
- Several `for` parts in one comprehension act as nested loops.
- Generators `(expr for x in data)` and `yield` functions produce values lazily, saving memory and allowing endless streams.
- `first, *rest = seq`, `a, b = b, a` and `zip(*rows)` are standard unpacking tricks.
- `itertools.combinations`, `permutations` and `product` enumerate libraries of building blocks.
- `any`, `all`, `Counter` and `defaultdict` shorten everyday checks, counts and groupings.
```

## Exercises

**★ 1.** Use a comprehension to make a list of wavelengths (nm) from wavenumbers `[1700, 2900, 3300]` cm⁻¹ using λ = 10⁷ / ν̃.

```{admonition} Solution
:class: dropdown
`[1e7 / nu for nu in [1700, 2900, 3300]]` gives about 5882, 3448 and 3030 nm.
```

**★★ 2.** Write a generator that yields the Balmer-series wavelengths $1/\lambda = R_H(1/4 - 1/n^2)$ for n = 3, 4, 5, …, stopping below 370 nm.

**★★★ 3.** Use `itertools.product` to enumerate all esters from five acids and five alcohols, then count unique ones with RDKit canonical SMILES.
