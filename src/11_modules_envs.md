---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Modules, Packages and Environments

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 1–10.
```

Python's real power is its ecosystem: hundreds of thousands of packages written by scientists for scientists. This chapter explains how to use them, how to share your own code and how to make your results reproducible.

## Importing

```{code-cell} ipython3
import math                       # whole module
from statistics import mean, stdev  # specific names
import numpy as np                # conventional alias

titres = [24.85, 24.90, 24.80, 24.95]
print(f"{mean(titres):.2f} ± {stdev(titres):.2f} mL")
print(np.sqrt(2), math.pi)
```

| Alias convention | Library |
|---|---|
| `import numpy as np` | arrays |
| `import pandas as pd` | tables |
| `import matplotlib.pyplot as plt` | plots |
| `import seaborn as sns` | statistical plots |
| `from rdkit import Chem` | molecules |

## The standard library: batteries included

```{code-cell} ipython3
import random, statistics, datetime, itertools, fractions, decimal
print(fractions.Fraction(2, 6))                # exact stoichiometric ratios
print(decimal.Decimal("0.1") + decimal.Decimal("0.2"))   # exact decimal arithmetic
random.seed(0)
print(random.choice(["He", "Ne", "Ar"]))
print(datetime.date.today().isoformat())
```

## Inspecting a module

```{code-cell} ipython3
import rdkit
from rdkit.Chem import Descriptors
print(rdkit.__version__)
print(len(Descriptors._descList), "descriptors available, e.g.", [d[0] for d in Descriptors._descList[:5]])
```

`help(module)` and `dir(module)` list what a module provides. In Jupyter, type a name followed by `?` to see its documentation.

## Writing your own module

Any `.py` file is a module. Let us write a tiny one, `chemtools.py`.

```{code-cell} ipython3
code = '''"""chemtools: a few handy chemistry functions."""
R = 8.314462618  # J/(mol K)

def kelvin(t_c):
    return t_c + 273.15

def ideal_gas_volume(n, T, P=101325):
    return n * R * T / P
'''
with open("chemtools.py", "w") as f:
    f.write(code)

import importlib, chemtools
importlib.reload(chemtools)
print(chemtools.ideal_gas_volume(1, chemtools.kelvin(0)) * 1000, "L")
```

A **package** is a folder of modules with an `__init__.py` file. Packages are shared on PyPI (`pip install ...`) and conda-forge (`conda install ...`).

## Installing packages

```bash
# conda (recommended for scientific stacks such as RDKit)
conda install -c conda-forge rdkit

# pip
pip install py3Dmol

# inside a notebook
%pip install pint
```

## Virtual environments

Different projects need different versions. An **environment** is an isolated set of packages.

```bash
conda create -n mdp python=3.12 rdkit numpy pandas matplotlib -c conda-forge
conda activate mdp
conda env export > environment.yml    # record exactly what you used
```

The book's own `environment.yml` creates everything you need with one command (see *Installing Python*).

## Reproducible research checklist

```{admonition} ⚗️ Chemistry note
:class: chem-note
A computational result is a measurement. Record the "instrument settings": package versions, random seeds and input files, just as you would record a spectrometer's slit width.
```

```{code-cell} ipython3
import sys, platform, numpy, pandas, matplotlib, sklearn
print("Python", sys.version.split()[0], "on", platform.system())
for lib in (numpy, pandas, matplotlib, sklearn, rdkit):
    print(f"{lib.__name__:12s} {lib.__version__}")
```

1. Keep code in version control (git and GitHub).
2. Pin package versions in `environment.yml`.
3. Set random seeds (`np.random.seed(42)`).
4. Never edit raw data; write code that transforms it.
5. Share notebooks that run top-to-bottom without errors.

## Exercises

**★ 1.** Use `statistics` to compute the mean and standard deviation of five melting-point readings.

**★★ 2.** Add a `pH(conc)` function to `chemtools.py`, reload the module and test it.

**★★★ 3.** Turn `chemtools` into a package folder with `gases.py` and `solutions.py`, and import from both.
