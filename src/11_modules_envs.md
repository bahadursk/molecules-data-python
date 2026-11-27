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

Python on its own knows only a few dozen built-in functions. Its real power is its ecosystem: hundreds of thousands of **packages** written by scientists for scientists, covering everything from arrays and plots to quantum chemistry and machine learning. Instead of writing a curve fitter or a SMILES parser yourself, you *import* one that thousands of people have already tested. This chapter explains how importing works, what the standard library already gives you, how to turn your own functions into a reusable module, how to install new packages, and how to keep a project's packages isolated and recorded so that your results can be reproduced by someone else, or by you next year.

## Importing

A **module** is a file of Python code that defines functions, classes and variables for other code to use. The `import` statement loads a module and makes its contents available. There are three common forms. `import math` loads the whole module, and you then write `math.sqrt` with the module's name in front, which makes it clear where each name came from. `from statistics import mean` takes specific names out of a module so you can use them directly. `import numpy as np` loads a module under a shorter **alias**, which saves typing for libraries you use on every line.

The most basic import gives you the module's name, and its contents are reached with a dot:

```{code-cell} ipython3
import math
print(math.pi)
print(math.sqrt(2))
```

All three forms can be mixed in one program:

```{code-cell} ipython3
import math                       # whole module
from statistics import mean, stdev  # specific names
import numpy as np                # conventional alias

titres = [24.85, 24.90, 24.80, 24.95]
print(f"{mean(titres):.2f} ± {stdev(titres):.2f} mL")
print(np.sqrt(2), math.pi)
```

The scientific community has agreed on standard aliases, and you should use them so that your code looks like everyone else's examples:

| Alias convention | Library |
|---|---|
| `import numpy as np` | arrays |
| `import pandas as pd` | tables |
| `import matplotlib.pyplot as plt` | plots |
| `import seaborn as sns` | statistical plots |
| `from rdkit import Chem` | molecules |

Avoid `from module import *`, which pours every name from a module into your notebook. It hides where names came from and can silently replace one of your own variables.

## The standard library: batteries included

The **standard library** is the large collection of modules that comes with every Python installation, so you can use it without installing anything. You have already met `math`, `re`, `csv`, `json`, `pathlib`, `itertools` and `collections`. A few more are useful in chemistry. `fractions` does exact arithmetic with ratios, which suits stoichiometry where 2/6 should stay exactly 1/3. `decimal` does exact decimal arithmetic, avoiding the binary rounding of Chapter 2 when you need it (for example in accounting for reagent stock). `random` generates pseudo-random numbers for simulations, and `datetime` handles dates and times.

```{code-cell} ipython3
import random, statistics, datetime, itertools, fractions, decimal
print(fractions.Fraction(2, 6))                # exact stoichiometric ratios
print(decimal.Decimal("0.1") + decimal.Decimal("0.2"))   # exact decimal arithmetic
random.seed(0)
print(random.choice(["He", "Ne", "Ar"]))
print(datetime.date.today().isoformat())
```

## Inspecting a module

When you meet a new module, you need to find out what it contains. The built-in `dir(module)` lists the names a module defines, and `help(name)` prints the documentation of any module, function or class. In Jupyter, typing a name followed by `?` (for example `math.log?`) shows the same documentation in a pop-up, and pressing Tab after a dot lists the available names. Most packages also record their version in `__version__`, which is worth checking when an example from the internet does not behave as described.

```{code-cell} ipython3
print([name for name in dir(math) if name.startswith("log")])
help(math.log10)
```

RDKit, for example, provides more than two hundred molecular descriptors in one module:

```{code-cell} ipython3
import rdkit
from rdkit.Chem import Descriptors
print(rdkit.__version__)
print(len(Descriptors._descList), "descriptors available, e.g.", [d[0] for d in Descriptors._descList[:5]])
```

## Writing your own module

Any file ending in `.py` is a module, and its name (without `.py`) is what you import. Moving functions you use often out of a notebook and into a module means you write them once and import them into every analysis, and when you fix a bug, every notebook gets the fix. The module must be in the same folder as your notebook, or somewhere else Python knows to look. Let us write a tiny one, `chemtools.py`. Here the file is written from code for the sake of the example; normally you would create it in a text editor.

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
print(chemtools.R)
```

Python imports a module only once per session, so if you edit `chemtools.py` while a notebook is open, the notebook keeps using the old version. `importlib.reload(chemtools)` forces it to read the file again; restarting the kernel does the same.

A **package** is a folder of modules with an `__init__.py` file, which lets you organise a larger project into parts, imported as `from chemtools import gases`. Packages are shared with the world on two main sites: **PyPI**, the Python Package Index, from which `pip` installs, and **conda-forge**, a community collection used by `conda`.

## Installing packages

Packages that are not in the standard library must be installed once on your computer before you can import them. There are two main installers. `conda` installs packages together with any compiled, non-Python libraries they depend on, which makes it the recommended choice for the scientific stack, and especially for RDKit. `pip` installs from PyPI and has the largest selection. Inside a Jupyter notebook, the `%pip` command installs into the same Python that the notebook is running, which avoids the common mistake of installing into a different Python.

```bash
# conda (recommended for scientific stacks such as RDKit)
conda install -c conda-forge rdkit

# pip
pip install py3Dmol

# inside a notebook
%pip install pint
```

If an `import` fails with `ModuleNotFoundError`, the package is either not installed or installed into a different Python. You can check which Python your notebook uses with `sys.executable`:

```{code-cell} ipython3
import sys
print(sys.executable)
```

## Virtual environments

Different projects often need different versions of the same package: an old analysis may depend on an earlier pandas, while a new one needs the latest. Installing everything into one place leads to conflicts. An **environment** is an isolated folder containing its own Python and its own set of packages, so each project can have exactly what it needs without affecting the others. With conda you create an environment, *activate* it to make it the current one, and can export a list of its exact contents to a file so that anyone can recreate it.

```bash
conda create -n mdp python=3.12 rdkit numpy pandas matplotlib -c conda-forge
conda activate mdp
conda env export > environment.yml    # record exactly what you used
```

The book's own `environment.yml` creates everything you need with one command (see *Installing Python*).

## Reproducible research checklist

**Reproducibility** means that someone else, given your data and code, gets the same results. For computational chemistry this is not automatic: a different package version can change a default setting, and an unseeded random number generator gives a different train/test split every time. The cure is to record your computational "instrument settings" as carefully as you record experimental ones.

```{admonition} ⚗️ Chemistry note
:class: chem-note
A computational result is a measurement. Record the "instrument settings": package versions, random seeds and input files, just as you would record a spectrometer's slit width.
```

A cell like this at the end of a notebook records the versions that produced the results:

```{code-cell} ipython3
import sys, platform, numpy, pandas, matplotlib, sklearn
print("Python", sys.version.split()[0], "on", platform.system())
for lib in (numpy, pandas, matplotlib, sklearn, rdkit):
    print(f"{lib.__name__:12s} {lib.__version__}")
```

Seeding a random number generator makes "random" results repeat exactly, which you can check directly:

```{code-cell} ipython3
random.seed(42); first = [random.random() for _ in range(3)]
random.seed(42); second = [random.random() for _ in range(3)]
print(first == second)
```

1. Keep code in version control (git and GitHub).
2. Pin package versions in `environment.yml`.
3. Set random seeds (`np.random.default_rng(42)`, Chapter 12).
4. Never edit raw data; write code that transforms it.
5. Share notebooks that run top-to-bottom without errors.

```{admonition} Summary
:class: tip
- `import module`, `from module import name` and `import module as alias` load code; use the standard aliases (`np`, `pd`, `plt`).
- The standard library (`math`, `statistics`, `fractions`, `random`, `datetime`, …) needs no installation.
- `dir()`, `help()` and `name?` show what a module offers; `__version__` gives its version.
- Any `.py` file is a module; a folder with `__init__.py` is a package; `importlib.reload` picks up edits.
- Install with `conda` (scientific stack) or `pip`/`%pip`; isolate projects in environments and export them to `environment.yml`.
- Record versions and seeds so your results can be reproduced.
```

## Exercises

**★ 1.** Use `statistics` to compute the mean and standard deviation of five melting-point readings.

```{admonition} Solution
:class: dropdown
`from statistics import mean, stdev; mp = [122.1, 122.4, 121.9, 122.3, 122.0]; print(mean(mp), stdev(mp))`
```

**★★ 2.** Add a `pH(conc)` function to `chemtools.py`, reload the module and test it.

**★★★ 3.** Turn `chemtools` into a package folder with `gases.py` and `solutions.py`, and import from both.
