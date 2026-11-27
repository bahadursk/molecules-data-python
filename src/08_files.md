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

Instruments, simulation packages and databases all talk in files. In this chapter we read and write plain text, CSV, JSON, XYZ and SDF files, the formats you will meet most often.

## Paths with `pathlib`

```{code-cell} ipython3
from pathlib import Path

workdir = Path("chem_files")
workdir.mkdir(exist_ok=True)
print(workdir.resolve())
```

## Writing and reading text

The `with` statement opens a file and closes it automatically, even if an error happens.

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

Extract numbers and compute the percentage yield:

```{code-cell} ipython3
values = {}
for line in notebook.read_text().splitlines()[1:]:
    key, value = line.split(":")
    values[key.strip()] = float(value.split()[0])

M_sal, M_asp = 138.12, 180.16
theoretical = values["salicylic acid"] / M_sal * M_asp
print(f"theoretical {theoretical:.2f} g, yield {100 * values['crude yield'] / theoretical:.1f} %")
```

## CSV files

Comma-separated values are the universal language of spreadsheets and instruments.

```{code-cell} ipython3
import csv, math

spectrum = workdir / "uv_vis.csv"
with open(spectrum, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["wavelength_nm", "absorbance"])
    for wl in range(400, 701, 5):
        A = 0.85 * math.exp(-((wl - 520) / 30) ** 2) + 0.02
        writer.writerow([wl, round(A, 4)])

with open(spectrum) as f:
    reader = csv.DictReader(f)
    rows = [(int(r["wavelength_nm"]), float(r["absorbance"])) for r in reader]

wl_max, A_max = max(rows, key=lambda r: r[1])
print(f"{len(rows)} points, λmax = {wl_max} nm, A = {A_max}")
```

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

```{code-cell} ipython3
import json
compound = {"name": "caffeine", "formula": "C8H10N4O2", "mp_C": 235,
            "smiles": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "hazards": ["H302"]}
(workdir / "caffeine.json").write_text(json.dumps(compound, indent=2))
print((workdir / "caffeine.json").read_text())
print(json.loads((workdir / "caffeine.json").read_text())["mp_C"])
```

## XYZ coordinate files

An XYZ file lists the number of atoms, a comment line and then one line per atom.

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

## SDF files with RDKit

The SD (structure–data) format stores molecules with their 2D or 3D connection table plus any number of properties. RDKit reads and writes it directly.

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

```{code-cell} ipython3
mols = [m for m in Chem.SDMolSupplier(str(workdir / "drugs.sdf"))]
Draw.MolsToGridImage(mols, legends=[f"{m.GetProp('_Name')} ({m.GetProp('MolWt')})" for m in mols],
                     molsPerRow=4, subImgSize=(220, 180))
```

## A lab-data logger

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

## Exercises

**★ 1.** Write a list of five solvents and their boiling points to a CSV file, then read it back and print the lowest boiling one.

**★★ 2.** Write `write_xyz(path, atoms, comment)` that is the inverse of `read_xyz`, and round-trip the water molecule.

**★★★ 3.** Parse the reactor log and report the average heating rate in °C per minute using `datetime.fromisoformat`.
