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

Chemistry is full of text: compound names, molecular formulas, CAS registry numbers, sample labels and line notations such as SMILES. Python stores any piece of text as a **string** (type `str`), which is simply an ordered sequence of characters. Strings matter in scientific computing because a great deal of chemical information is *only* available as text, in file headers, database records and instrument exports, and you must be able to clean it, search it and pull numbers out of it before any calculation can start. This chapter shows how to create strings, how to reach individual characters, which built-in tools tidy and test text, and how a molecule can be encoded as a string that a program can read.

## Creating strings

You create a string by putting text between quotation marks. Single quotes `'...'` and double quotes `"..."` are identical in Python; having both lets you include one kind of quote inside the other. Triple quotes `"""..."""` allow a string to run over several lines, which is handy for notes and for the documentation strings of functions (Chapter 6). The quotes themselves are not part of the string; they only mark where it begins and ends.

```{code-cell} ipython3
name = "caffeine"
formula = 'C8H10N4O2'
label = "Sample 'A' from batch 3"     # single quotes inside double quotes
print(name)
print(formula)
print(label)
```

The function `len()` tells you how many characters a string contains, and `+` joins two strings into a new one (this is called *concatenation*). Multiplying a string by an integer repeats it.

```{code-cell} ipython3
print(len(name))
print(name + " is " + formula)
print("-" * 20)
```

Strings can be as long as you like, and a molecule's SMILES string is just another string:

```{code-cell} ipython3
smiles = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
note = """Multi-line strings
are written with triple quotes."""
print(name, formula, smiles, sep=" | ")
print(len(smiles), "characters in the SMILES string")
print(note)
```

Strings are **immutable**: once created, a string cannot be changed in place. Every operation that seems to modify one, such as replacing a word, actually returns a *new* string and leaves the original untouched. This makes strings safe to share between parts of a program, and it means you must store the result if you want to keep it.

## Indexing and slicing

Because a string is a sequence, every character has a numbered position called its **index**. Python starts counting at **0**, so the first character is at index 0, the second at 1, and so on. Negative indices count backwards from the end, with −1 the last character. You read a character with square brackets: `text[0]`.

```{code-cell} ipython3
element = "Chlorine"
print(element[0])     # first character
print(element[1])     # second character
print(element[-1])    # last character
```

A **slice** `text[start:stop]` extracts a range of characters, from `start` up to *but not including* `stop`. Leaving out `start` means "from the beginning" and leaving out `stop` means "to the end". An optional third number gives a step, so `[::-1]` walks backwards through the whole string. The "up to but not including" rule seems odd at first, but it means `text[:2]` always gives exactly two characters and `text[:2] + text[2:]` always rebuilds the original.

```{code-cell} ipython3
print(element[0:2])      # characters 0 and 1
print(element[:2] + element[2:])
print(element[0] + element[3], "is the real symbol")
print(element[::-1])     # reversed
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Element symbols are not always the first letters of the English name: Na (natrium), K (kalium), Fe (ferrum), Pb (plumbum), W (wolfram).
```

Asking for an index beyond the end gives an `IndexError`, but slices are forgiving and simply stop at the end, so `element[5:100]` is `"ine"`.

## Useful string methods

A **method** is a function that belongs to a particular kind of object and is called with a dot after the object: `text.upper()`. Strings come with dozens of methods, and a small number of them do most of the work of cleaning real-world text. `strip()` removes spaces and line breaks from both ends, which matters because text copied from spreadsheets or read from files often carries invisible whitespace that makes `"NaCl " == "NaCl"` false. `lower()`, `upper()` and `title()` change letter case, which lets you compare names regardless of how they were typed.

```{code-cell} ipython3
raw = "  Sodium Chloride  "
clean = raw.strip()
print("[" + raw + "]")
print("[" + clean + "]")
print(clean.lower(), clean.upper(), clean.title(), sep=" / ")
```

`replace(old, new)` swaps every occurrence of one piece of text for another. `split()` breaks a string into a list of pieces (Chapter 4 covers lists), splitting on whitespace by default or on any separator you give it, and `join()` does the reverse, gluing a list of strings together with a separator. `count()` counts occurrences and `startswith()` / `endswith()` test the beginning or end.

```{code-cell} ipython3
print(clean.replace("Chloride", "Bromide"))
print(clean.split())                       # split on whitespace
print("25.0,0.105,7.2".split(","))         # split a line of comma-separated values
print("-".join(["H", "O", "H"]))
print("CCO".count("C"), "carbons in ethanol SMILES")
print("c1ccccc1".startswith("c1"), "aromatic ring?")
```

The `in` operator asks whether one string appears inside another and gives a Boolean. It is the quickest way to search a name or a label:

```{code-cell} ipython3
print("chlor" in "2-chlorophenol")
print("Br" in "C6H5Cl")
```

## Checking content

String methods whose names start with `is` test what kind of characters a string contains and return `True` or `False`. `isdigit()` is true only if every character is a digit, `isalpha()` if every character is a letter, and `isupper()` / `islower()` check case. These tests are used to validate input before converting it, so a program can reject a bad value with a clear message instead of crashing later.

```{code-cell} ipython3
print("2024".isdigit(), "20.24".isdigit())
print("Na".isalpha(), "Na2".isalpha())
print("N".isupper(), "a".islower())
```

A real example is the CAS registry number, the identifier that Chemical Abstracts Service gives each substance. Its last digit is a **check digit**: multiply the other digits, read from right to left, by 1, 2, 3 and so on, add the products, and the last digit of the total must equal the check digit. A typo in any single digit breaks the check, which is why databases use it.

```{code-cell} ipython3
cas = "58-08-2"           # caffeine CAS number
digits = cas.replace("-", "")
print(digits, digits.isdigit())

# The last digit of a CAS number is a checksum
body, check = digits[:-1], int(digits[-1])
total = sum((i + 1) * int(d) for i, d in enumerate(reversed(body)))
print("weighted sum:", total, "-> last digit", total % 10, "check digit", check)
print("valid CAS:", total % 10 == check)
```

The line computing `total` uses `enumerate` and a generator expression, both explained in Chapter 7; read it as "for each digit `d` at position `i` counted from the right, add `(i + 1) × d`".

## Formatting numbers inside text

Most output in science is a sentence with numbers in it. An **f-string**, a string with the letter `f` before the opening quote, lets you put any expression inside curly braces and have its value inserted. After the expression, a colon introduces a *format specification* that controls how the number looks: `.3f` means three digits after the decimal point, `.2e` means scientific notation with two decimals, and `8.3f` reserves eight characters of width so that columns line up.

```{code-cell} ipython3
M = 194.19
n = 0.0025
print(f"Molar mass {M} g/mol")
print(f"Moles {n:.4f} mol, or {n:.2e} mol")
print(f"Mass {n * M:.3f} g")
for name_, mm in [("water", 18.015), ("glucose", 180.156), ("caffeine", 194.19)]:
    print(f"{name_:10s}{mm:8.2f}")
```

`10s` pads a string to ten characters, so the numbers form a tidy column. You will use f-strings in almost every chapter.

## Parsing a chemical formula by hand

**Parsing** means reading text according to a set of rules and turning it into a structured result. Counting atoms in a formula like `C6H12O6` is a good first parser, because the rules are short: an uppercase letter starts a new element symbol, a lowercase letter right after it continues the same symbol (as in `Na`), and the digits that follow give the count, with no digits meaning one atom. We step through the formula one character at a time, keeping our position in `i`.

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

This uses dictionaries (Chapter 4), loops (Chapter 5) and functions (Chapter 6). Come back to it after those chapters and it will read easily. The expression `int(number or 1)` is a small idiom: an empty string counts as false, so when no digits were found the `or` supplies 1 instead.

## Regular expressions

Writing a parser character by character works, but it is long. A **regular expression** (regex) is a compact pattern that describes the shape of the text you are looking for, and the `re` module searches text for matches. The pattern language takes some practice, but a few symbols go a long way: `[A-Z]` means any one uppercase letter, `[a-z]?` means an optional lowercase letter, `\d` means a digit and `*` means "zero or more of the previous thing". Parentheses mark the parts you want returned. The `r` before the pattern makes it a *raw string*, so that backslashes are passed to `re` unchanged.

A very basic pattern finds every number in a line of text, which is useful for pulling values out of instrument logs:

```{code-cell} ipython3
import re
line = "Run 7: T = 298.15 K, P = 101.3 kPa, yield 87 %"
print(re.findall(r"\d+\.?\d*", line))
```

The pattern `([A-Z][a-z]?)(\d*)` means "an uppercase letter, an optional lowercase letter, then any digits", which is the formula rule from the previous section in one line:

```{code-cell} ipython3
print(re.findall(r"([A-Z][a-z]?)(\d*)", "C6H12O6"))
print(re.findall(r"([A-Z][a-z]?)(\d*)", "Ca3(PO4)2"))
```

Brackets are not handled yet: the `2` after `(PO4)` is lost. Exercise 3 asks you to fix that.

## Pretty formulas with Unicode

Text in Python is Unicode, so it can contain subscript and superscript digits, Greek letters and arrows as easily as plain letters. You can type them directly (`"H₂O"`, `"ΔG"`), or convert plain text automatically. The method `str.maketrans` builds a translation table that maps each character to a replacement, and `translate()` applies the table to a string.

```{code-cell} ipython3
print("H₂O", "ΔG° = −237 kJ mol⁻¹", "A → B")
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
SUP = str.maketrans("0123456789+-", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻")
print("C8H10N4O2".translate(SUB))
print("Fe" + "3+".translate(SUP), "SO" + "4".translate(SUB) + "2-".translate(SUP))
```

## SMILES are strings

A molecular formula tells you which atoms a molecule contains but not how they are connected: ethanol and dimethyl ether are both C₂H₆O. **SMILES** (Simplified Molecular Input Line Entry System) solves this by writing the connectivity as a line of text. Atoms are written as their symbols, neighbouring atoms are bonded, `=` marks a double bond, parentheses mark a branch, matching digits mark the two ends of a ring closure, and lowercase atoms are aromatic. Hydrogens are usually left implicit. Because a SMILES is just a string, everything in this chapter applies to it: you can store it, slice it, search it and print it.

```{code-cell} ipython3
print("ethanol:", "CCO", "  dimethyl ether:", "COC")
```

Here is a small collection stored in a dictionary (Chapter 4), with a crude ring count from the ring-closure digits:

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

Counting characters is fragile: it fails for molecules with ring-closure digits other than 1 and for atoms such as `Cl` that contain a `C`. In Chapters 19 and 20 RDKit will turn these strings into real molecule objects that understand chemistry, which we can already draw:

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Draw
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in molecules.values()],
                     legends=list(molecules), molsPerRow=4, subImgSize=(200, 160))
```

```{admonition} Summary
:class: tip
- A string is an immutable sequence of characters, written in single, double or triple quotes.
- Index with `s[0]` and `s[-1]`, slice with `s[start:stop]` (the stop is excluded).
- `strip`, `lower`, `replace`, `split` and `join` clean and reshape text; `in` searches it; `is...` methods test it.
- f-strings such as `f"{x:.3f}"` put formatted numbers into text.
- Regular expressions describe patterns compactly; `re.findall` returns every match.
- SMILES encodes a molecule's connectivity as a string, which RDKit can turn into a molecule object.
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
