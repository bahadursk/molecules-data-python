---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Reading and Writing Chemical Files

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 1–7.
```

Instruments, simulation packages and databases all communicate through **files**. A UV–vis spectrometer exports a table of wavelengths and absorbances, a quantum-chemistry program writes atomic coordinates, and a compound database hands you thousands of structures in one download. A variable in Python lives only while your notebook runs; a file stays on disk, so it is how data enters your analysis and how your results outlive it. Being able to read and write files yourself means you are never stuck copying numbers by hand, and you can process a hundred files as easily as one. This chapter covers the formats you will meet most often, plain text, CSV, JSON, XYZ and SDF, together with the tools for finding files and opening them safely.

## Paths with `pathlib`

A **path** is the address of a file or folder on your computer, such as `chem_files/water.xyz`. Paths are written differently on Windows (`\`) and on macOS or Linux (`/`), and building them by gluing strings together is a common source of bugs. The standard-library module `pathlib` represents a path as a `Path` object that knows the rules of your operating system. You join path pieces with the `/` operator, ask questions such as `.exists()` or `.suffix`, and create folders with `.mkdir()`. A *relative* path like `chem_files` is interpreted from the folder your notebook is running in; `.resolve()` shows the full *absolute* path.

```{code-cell} ipython3
from pathlib import Path

workdir = Path("chem_files")
workdir.mkdir(exist_ok=True)        # exist_ok: no error if it is already there
print(workdir.resolve())
```

Joining and inspecting paths needs no string handling at all:

```{code-cell} ipython3
p = workdir / "spectra" / "sample_01.csv"
print(p)
print(p.name, p.stem, p.suffix)     # file name, name without extension, extension
print(p.parent)                     # the folder that contains it
print(p.exists())                   # nothing written there yet
```

## Writing and reading text

Every file is opened in a **mode**: `"r"` to read (the default), `"w"` to write (which creates the file or *erases* an existing one), and `"a"` to append to the end. Opening a file ties up a resource of the operating system, and the file must be closed again so that everything is actually written to disk. The `with` statement handles this for you: the file is open inside the indented block and is closed automatically when the block ends, even if an error happens halfway through. Always use `with` when you open files.

The simplest round trip writes one line and reads it back. `"\n"` is the newline character that ends a line; `write` does not add it for you.

```{code-cell} ipython3
note = workdir / "hello.txt"
with open(note, "w") as f:
    f.write("NaCl dissolves in water\n")

with open(note) as f:
    print(f.read())
```

A lab notebook is a more realistic text file, with one fact per line. Looping over an open file gives one line at a time, which works for files of any size, and `rstrip()` removes the trailing newline before printing.

```{code-cell} ipython3
notebook = workdir / "lab_notebook.txt"
with open(notebook, "w") as f:
    f.write("2026-11-27 Synthesis of aspirin\n")
    f.write("salicylic acid: 2.012 g\n")
    f.write("acetic anhydride: 5.0 mL\n")
    f.write("crude yield: 2.31 g\n")

with open(notebook) as f:
    for line in f:
        print(line.rstrip())
```

For small files, `Path.read_text()` and `Path.write_text()` do the opening and closing in one call. Here each line after the date is split at the colon into a name and a value, and the number is taken from the value with `split()[0]`, which drops the unit. The numbers then feed a percentage-yield calculation, since one mole of salicylic acid gives one mole of aspirin.

```{code-cell} ipython3
values = {}
for line in notebook.read_text().splitlines()[1:]:
    key, value = line.split(":")
    values[key.strip()] = float(value.split()[0])
print(values)

M_sal, M_asp = 138.12, 180.16
theoretical = values["salicylic acid"] / M_sal * M_asp
print(f"theoretical {theoretical:.2f} g, yield {100 * values['crude yield'] / theoretical:.1f} %")
```

Keep in mind that this is a *crude* yield: the product is still damp and contains some acetic acid, so the yield after drying and recrystallisation will be lower.

## CSV files

**CSV** (comma-separated values) is a plain-text table: each line is a row, and commas separate the columns. The first row usually holds the column names. CSV is the universal exchange format of spreadsheets and instruments because any program can read it. You could split each line at the commas yourself, but the standard `csv` module handles the awkward cases, such as a compound name that itself contains a comma. `csv.writer` writes rows from lists, and `csv.DictReader` reads each row as a dictionary keyed by the column names, which makes the code self-describing. Opening with `newline=""` stops blank lines appearing between rows on Windows.

Values read from a CSV file are always strings, so they must be converted with `int()` or `float()` (Chapter 2). Here we simulate a UV–vis spectrum with a Gaussian band at 520 nm, write it to a file, and read it back:

```{code-cell} ipython3
import csv, math

spectrum = workdir / "uv_vis.csv"
with open(spectrum, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["wavelength_nm", "absorbance"])
    for wl in range(400, 701, 5):
        A = 0.85 * math.exp(-((wl - 520) / 30) ** 2) + 0.02
        writer.writerow([wl, round(A, 4)])

print(spectrum.read_text()[:80])     # the raw text of the first few lines

with open(spectrum) as f:
    reader = csv.DictReader(f)
    rows = [(int(r["wavelength_nm"]), float(r["absorbance"])) for r in reader]

wl_max, A_max = max(rows, key=lambda r: r[1])
print(f"{len(rows)} points, λmax = {wl_max} nm, A = {A_max}")
```

Colouring each slice of the band by its wavelength shows why this dye looks magenta: it absorbs green light, so we see the complementary colour.

```{code-cell} ipython3
import matplotlib.pyplot as plt
from matplotlib import cm
wls, As = zip(*rows)
plt.figure(figsize=(7, 3.5))
for i in range(len(wls) - 1):   # fill under the curve with the colour of each wavelength
    plt.fill_between(wls[i:i+2], As[i:i+2], color=cm.nipy_spectral((wls[i] - 380) / 400), alpha=0.8)
plt.plot(wls, As, "k", lw=1)
plt.xlabel("Wavelength / nm"); plt.ylabel("Absorbance")
plt.title("A dye absorbing green light looks magenta")
plt.show()
```

In Chapter 15 pandas will read such a file in a single line: `pd.read_csv("uv_vis.csv")`.

## JSON for structured data

CSV suits flat tables, but some records are nested: a compound has a name, a formula, a melting point and a *list* of hazard codes. **JSON** (JavaScript Object Notation) is a text format that stores exactly the structures you already know from Chapter 4, dictionaries, lists, strings, numbers and Booleans, nested to any depth. It is what most web databases, including PubChem (Chapter 24), return. The `json` module converts in both directions: `json.dumps` turns a Python object into JSON text, and `json.loads` turns JSON text back into Python objects. `indent=2` makes the text readable for humans.

```{code-cell} ipython3
import json
compound = {"name": "caffeine", "formula": "C8H10N4O2", "mp_C": 235,
            "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "hazards": ["H302"]}
(workdir / "caffeine.json").write_text(json.dumps(compound, indent=2))
print((workdir / "caffeine.json").read_text())

loaded = json.loads((workdir / "caffeine.json").read_text())
print(loaded["mp_C"], type(loaded["mp_C"]).__name__)
```

Unlike CSV, JSON keeps types: the melting point comes back as an integer, not a string.

## XYZ coordinate files

The **XYZ** format is the simplest way to store a molecule's 3D geometry and is written by nearly every computational chemistry program. The first line is the number of atoms, the second is a free comment, and each following line gives an element symbol and its x, y and z coordinates, usually in ångströms. Because the layout is fixed, a reader is only a few lines of code: read the atom count, skip the comment, and split each atom line into four pieces.

```{code-cell} ipython3
xyz_text = """3
water, B3LYP/6-31G* geometry
O   0.000000   0.000000   0.117300
H   0.000000   0.757200  -0.469200
H   0.000000  -0.757200  -0.469200
"""
(workdir / "water.xyz").write_text(xyz_text)

def read_xyz(path):
    lines = Path(path).read_text().splitlines()
    n = int(lines[0])
    atoms = []
    for line in lines[2:2 + n]:
        sym, x, y, z = line.split()
        atoms.append((sym, float(x), float(y), float(z)))
    return lines[1], atoms

comment, atoms = read_xyz(workdir / "water.xyz")
print(comment)
print(atoms[0])
```

With coordinates in hand, geometry follows. The distance between two atoms is the length of the vector between them, and the angle at a central atom comes from the dot product of the two bond vectors, $\cos\theta = \frac{\mathbf{a}\cdot\mathbf{b}}{|\mathbf{a}||\mathbf{b}|}$. `math.dist` gives the distance directly, and `math.hypot` gives a vector's length.

```{code-cell} ipython3
def distance(a, b):
    return math.dist(a[1:], b[1:])

def angle(a, b, c):
    ab = [p - q for p, q in zip(a[1:], b[1:])]
    cb = [p - q for p, q in zip(c[1:], b[1:])]
    dot = sum(p * q for p, q in zip(ab, cb))
    return math.degrees(math.acos(dot / (math.hypot(*ab) * math.hypot(*cb))))

O, H1, H2 = atoms
print(f"O–H = {distance(O, H1):.3f} Å, H–O–H = {angle(H1, O, H2):.1f}°")
```

Both numbers are close to the experimental geometry of water (0.96 Å and 104.5°), a quick check that the file was read correctly.

## SDF files with RDKit

XYZ files store positions but no bonds and no properties. The **SD file** (structure–data file, extension `.sdf`) stores, for each molecule, a *connection table* listing atoms and bonds, followed by any number of named data fields such as a molecular weight or a measured activity. Molecules are separated by a line of four dollar signs, `$$$$`. Because one SDF can hold thousands of molecules with their data, it is the standard format for compound libraries and database downloads. RDKit writes SDF with `SDWriter` and reads it with `SDMolSupplier`; `SetProp` attaches a data field, and the special property `_Name` becomes the molecule's title line.

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw

drugs = {"aspirin": "CC(=O)Oc1ccccc1C(=O)O", "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
         "paracetamol": "CC(=O)Nc1ccc(O)cc1", "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"}

with Chem.SDWriter(str(workdir / "drugs.sdf")) as w:
    for name, smi in drugs.items():
        m = Chem.MolFromSmiles(smi)
        m.SetProp("_Name", name)
        m.SetProp("MolWt", f"{Descriptors.MolWt(m):.2f}")
        w.write(m)

print((workdir / "drugs.sdf").read_text()[:600], "...")
```

Reading the file back recovers the molecules with their properties, ready to draw or analyse:

```{code-cell} ipython3
mols = [m for m in Chem.SDMolSupplier(str(workdir / "drugs.sdf"))]
Draw.MolsToGridImage(mols, legends=[f"{m.GetProp('_Name')} ({m.GetProp('MolWt')})" for m in mols],
                     molsPerRow=4, subImgSize=(220, 180))
```

## Processing many files at once

A real experiment produces a folder of files, one per sample. `Path.glob(pattern)` finds every file whose name matches a pattern, where `*` stands for any characters, so `glob("*.csv")` lists all CSV files in a folder. Combined with a loop, this turns a tedious afternoon of opening files into a few seconds.

```{code-cell} ipython3
for i, conc in enumerate([0.1, 0.2, 0.4], start=1):
    (workdir / f"std_{i}.csv").write_text(f"conc_M,absorbance\n{conc},{0.9 * conc + 0.01:.3f}\n")

for path in sorted(workdir.glob("std_*.csv")):
    with open(path) as f:
        row = next(csv.DictReader(f))        # first (and only) data row
    print(path.name, row)
```

## A lab-data logger

Programs often *write* files as well, for example to log readings from a reactor. Timestamps should be written in a standard form so that other programs can read them back; `datetime.isoformat()` produces the international ISO 8601 format, such as `2026-11-27T09:05:00`. Here the `random` module adds small Gaussian noise to simulated temperature and pressure readings taken every five minutes, and `random.seed(1)` makes the "random" numbers the same every time the notebook runs, so the results are reproducible.

```{code-cell} ipython3
from datetime import datetime, timedelta
import random

random.seed(1)
log = workdir / "reactor_log.csv"
start = datetime(2026, 11, 27, 9, 0)
with open(log, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["time", "T_C", "P_bar"])
    for i in range(12):
        w.writerow([(start + timedelta(minutes=5 * i)).isoformat(),
                    round(60 + 0.8 * i + random.gauss(0, 0.3), 2),
                    round(1.0 + 0.02 * i + random.gauss(0, 0.005), 3)])
print(log.read_text())
```

```{admonition} Summary
:class: tip
- Build and inspect paths with `pathlib.Path`; join with `/`, find files with `.glob()`.
- Open files with `with open(path, mode) as f:` so they always close; `"w"` overwrites, `"a"` appends.
- Text read from files is a string: convert numbers with `float()` or `int()`.
- CSV stores flat tables (`csv.writer`, `csv.DictReader`); JSON stores nested data and keeps types (`json.dumps`, `json.loads`).
- XYZ stores 3D coordinates; SDF stores molecules with bonds and data fields and is read and written by RDKit.
- Write timestamps with `isoformat()` and seed random numbers for reproducible files.
```

## Exercises

**★ 1.** Write a list of five solvents and their boiling points to a CSV file, then read it back and print the lowest boiling one.

**★★ 2.** Write `write_xyz(path, atoms, comment)` that is the inverse of `read_xyz`, and round-trip the water molecule.

**★★★ 3.** Parse the reactor log and report the average heating rate in °C per minute using `datetime.fromisoformat`.
