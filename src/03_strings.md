---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Strings: Formulas, Names and SMILES

```{admonition} 🟢 Beginner · school & first-year
:class: level-1
Builds on Chapters 1–2.
```

Chemistry is full of text: names, formulas, CAS numbers and line notations such as SMILES. Python stores text in **strings**.

## Creating strings

```{code-cell} ipython3
name = "caffeine"
formula = 'C8H10N4O2'
smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
note = """Multi-line strings
are written with triple quotes."""
print(name, formula, smiles, sep=" | ")
print(len(smiles), "characters in the SMILES string")
```

## Indexing and slicing

Each character has a position, starting at **0**. Negative indices count from the end.

```{code-cell} ipython3
element = "Chlorine"
print(element[0], element[1], element[-1])
print(element[0:2])      # characters 0 and 1
print(element[0] + element[3], "is the real symbol")
print(element[::-1])     # reversed
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Element symbols are not always the first letters of the English name: Na (natrium), K (kalium), Fe (ferrum), Pb (plumbum), W (wolfram).
```

## Useful string methods

```{code-cell} ipython3
raw = "  Sodium Chloride  "
clean = raw.strip()
print(clean.lower(), clean.upper(), clean.title(), sep=" / ")
print(clean.replace("Chloride", "Bromide"))
print(clean.split())                       # split on whitespace
print("-".join(["H", "O", "H"]))
print("CCO".count("C"), "carbons in ethanol SMILES")
print("c1ccccc1".startswith("c1"), "aromatic ring?")
```

## Checking content

```{code-cell} ipython3
cas = "58-08-2"           # caffeine CAS number
digits = cas.replace("-", "")
print(digits.isdigit())

# The last digit of a CAS number is a checksum
body, check = digits[:-1], int(digits[-1])
total = sum((i + 1) * int(d) for i, d in enumerate(reversed(body)))
print("valid CAS:", total % 10 == check)
```

## Parsing a chemical formula by hand

Let us count atoms in a formula like `C6H12O6`. We step through the characters: an uppercase letter starts a new element, a lowercase letter continues it, and digits give the count.

```{code-cell} ipython3
def parse_formula(formula):
    counts = {}
    i = 0
    while i < len(formula):
        symbol = formula[i]
        i += 1
        if i < len(formula) and formula[i].islower():
            symbol += formula[i]
            i += 1
        number = ""
        while i < len(formula) and formula[i].isdigit():
            number += formula[i]
            i += 1
        counts[symbol] = counts.get(symbol, 0) + int(number or 1)
    return counts

for f in ("C6H12O6", "NaCl", "Fe2O3", "C8H10N4O2"):
    print(f, parse_formula(f))
```

This uses dictionaries (Chapter 4), loops (Chapter 5) and functions (Chapter 6). Come back to it after those chapters and it will read easily.

## Regular expressions

The `re` module matches patterns in text. The pattern `([A-Z][a-z]?)(\d*)` means "an uppercase letter, an optional lowercase letter, then any digits".

```{code-cell} ipython3
import re
print(re.findall(r"([A-Z][a-z]?)(\d*)", "Ca3(PO4)2"))
```

Brackets are not handled yet. Exercise 3 asks you to fix that.

## Pretty formulas with Unicode

```{code-cell} ipython3
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
print("C8H10N4O2".translate(SUB))
print("Fe" + "3+".translate(SUP), "SO" + "4".translate(SUB) + "2-".translate(SUP))
```

## SMILES are strings

SMILES (Simplified Molecular Input Line Entry System) writes a molecule as text: atoms as symbols, `=` for double bonds, brackets for branches and digits for ring closures. Lowercase atoms are aromatic.

```{code-cell} ipython3
molecules = {
    "ethanol": "CCO",
    "acetic acid": "CC(=O)O",
    "benzene": "c1ccccc1",
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O",
}
for name, smi in molecules.items():
    print(f"{name:12s} {smi:24s} rings: {smi.count('1') // 2}")
```

Counting characters is fragile. In Chapter 20 RDKit will turn these strings into real molecule objects that we can draw:

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Draw
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in molecules.values()],
                     legends=list(molecules), molsPerRow=4, subImgSize=(200, 160))
```

## Exercises

**★ 1.** Convert `"sodium hydrogen carbonate"` to title case and count its words.

```{admonition} Solution
:class: dropdown
`s.title()` and `len(s.split())` (4 words).
```

**★★ 2.** Use `parse_formula` to compute the molar mass of `C8H10N4O2` with the masses C 12.011, H 1.008, N 14.007, O 15.999.

```{admonition} Solution
:class: dropdown
`sum(masses[el] * n for el, n in parse_formula("C8H10N4O2").items())` gives 194.19 g/mol.
```

**★★★ 3.** Extend `parse_formula` to handle brackets such as `Ca3(PO4)2`. Hint: parse the inside of a bracket recursively and multiply by the number after it.
