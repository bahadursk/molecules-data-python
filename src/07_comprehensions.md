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

"Pythonic" code is short, clear and uses the language's built-in tools. This chapter covers the idioms you will see in every scientific library.

## List comprehensions

A comprehension builds a list in one readable line.

```{code-cell} ipython3
temps_C = [-78.5, 0, 25, 37, 100]
temps_K = [t + 273.15 for t in temps_C]
print(temps_K)
```

The general form is `[expression for item in iterable if condition]`.

```{code-cell} ipython3
atomic_numbers = range(1, 37)
# Elements in the s-block of period 4 or noble gases
noble = [z for z in atomic_numbers if z in (2, 10, 18, 36)]
squares = [n**2 for n in range(1, 5)]
print(noble, squares)
```

## Dictionary and set comprehensions

```{code-cell} ipython3
import math
pKa = {"acetic": 4.76, "formic": 3.75, "benzoic": 4.20, "phenol": 9.95, "HF": 3.17}
Ka = {acid: 10 ** -p for acid, p in pKa.items()}
strong_ish = {acid for acid, p in pKa.items() if p < 4}
print({k: f"{v:.2e}" for k, v in Ka.items()})
print(strong_ish)
```

## Nested comprehensions: building orbitals

```{code-cell} ipython3
orbitals = [f"{n}{'spdf'[l]}" for n in range(1, 5) for l in range(n)]
print(orbitals)
```

The **Aufbau (n + l) rule** orders orbitals by n + l, then by n:

```{code-cell} ipython3
aufbau = sorted(((n, l) for n in range(1, 8) for l in range(min(n, 4))), key=lambda o: (o[0] + o[1], o[0]))
capacity = [2 * (2 * l + 1) for n, l in aufbau]

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

## Generators: lazy sequences

A generator produces values one at a time instead of building a whole list. Use round brackets.

```{code-cell} ipython3
total_mass = sum(m * n for m, n in [(12.011, 6), (1.008, 6)])   # benzene
print(total_mass)
```

Generator **functions** use `yield`. Here is an endless stream of radioactive-decay amounts:

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

## Unpacking tricks

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

```{code-cell} ipython3
from itertools import combinations, product, permutations

halides = ["F", "Cl", "Br", "I"]
print("Interhalogen pairs:", ["".join(p) for p in combinations(halides, 2)])

r_groups = ["H", "CH3", "OH", "Cl"]
library = list(product(r_groups, repeat=2))
print(len(library), "disubstituted benzenes (before symmetry):", library[:6], "...")
```

The library above can be turned into real molecules with RDKit:

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

```{code-cell} ipython3
# any / all
pHs = [6.8, 7.1, 7.4, 7.0]
print(all(6.5 < p < 7.5 for p in pHs), any(p > 7.3 for p in pHs))

# Conditional expression
T = 400
phase = "gas" if T > 373.15 else "liquid"
print(phase)

# Counting with collections.Counter
from collections import Counter, defaultdict
print(Counter("CC(=O)Oc1ccccc1C(=O)O".upper().replace("(", "").replace(")", "")))

groups = defaultdict(list)
for el, block in [("Na", "s"), ("Fe", "d"), ("C", "p"), ("K", "s"), ("Cu", "d")]:
    groups[block].append(el)
print(dict(groups))
```

## Exercises

**★ 1.** Use a comprehension to make a list of wavelengths (nm) from wavenumbers `[1700, 2900, 3300]` cm⁻¹ using λ = 10⁷ / ν̃.

**★★ 2.** Write a generator that yields the Balmer-series wavelengths $1/\lambda = R_H(1/4 - 1/n^2)$ for n = 3, 4, 5, …, stopping below 370 nm.

**★★★ 3.** Use `itertools.product` to enumerate all esters from five acids and five alcohols, then count unique ones with RDKit canonical SMILES.
