---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# 3D Structures, Conformers and py3Dmol

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 19–21.
```

Molecules are three-dimensional and flexible. Their shape decides how they pack in crystals, bind to proteins and smell. In this chapter we generate 3D coordinates, optimise them with force fields, explore conformers and view everything interactively with **py3Dmol**.

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdMolAlign, Descriptors3D
import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
```

## From SMILES to 3D

```{code-cell} ipython3
def to_3d(smiles, seed=42):
    """SMILES -> 3D molecule with hydrogens, embedded with ETKDG and relaxed with MMFF94."""
    m = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    AllChem.EmbedMolecule(m, params)
    AllChem.MMFFOptimizeMolecule(m)
    return m

caffeine = to_3d("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
print(Chem.MolToXYZBlock(caffeine)[:300])
```

**ETKDG** combines distance geometry with torsion preferences learned from crystal structures; **MMFF94** then relaxes the geometry.

## Interactive 3D viewing

Drag to rotate, scroll to zoom.

```{code-cell} ipython3
def show(mol, style="stick", width=420, height=320, surface=False):
    v = py3Dmol.view(width=width, height=height)
    v.addModel(Chem.MolToMolBlock(mol), "mol")
    v.setStyle({style: {}} if style != "ballstick" else {"stick": {"radius": 0.15}, "sphere": {"scale": 0.25}})
    if surface:
        v.addSurface(py3Dmol.VDW, {"opacity": 0.6, "colorscheme": "whiteCarbon"})
    v.setBackgroundColor("white")
    v.zoomTo()
    return v

show(caffeine, "ballstick").show()
```

```{code-cell} ipython3
show(to_3d("CC(=O)Oc1ccccc1C(=O)O"), "stick", surface=True).show()
```

### Several molecules side by side

```{code-cell} ipython3
fullerene = Chem.MolFromSmiles("c12c3c4c5c1c1c6c7c2c2c8c3c3c9c4c4c%10c5c5c1c1c6c6c%11c7c2c2c7c8c3c3c8c9c4c4c9c%10c5c5c1c1c6c6c%11c2c2c7c3c3c8c4c4c9c5c1c1c6c2c3c41")
fullerene = Chem.AddHs(fullerene)
AllChem.EmbedMolecule(fullerene, randomSeed=1)
v = py3Dmol.view(width=700, height=320, viewergrid=(1, 2), linked=False)
v.addModel(Chem.MolToMolBlock(fullerene), "mol", viewer=(0, 0))
v.setStyle({"stick": {"colorscheme": "cyanCarbon"}}, viewer=(0, 0))
macrocycle = to_3d("C1CCCCCCCCCCCCCCCCC1")    # an 18-membered macrocycle
v.addModel(Chem.MolToMolBlock(macrocycle), "mol", viewer=(0, 1))
v.setStyle({"stick": {"colorscheme": "magentaCarbon"}}, viewer=(0, 1))
v.zoomTo(); v.show()
```

## Geometry: bonds, angles and torsions

```{code-cell} ipython3
ethane = to_3d("CC")
conf = ethane.GetConformer()
print(f"C–C  {rdMolTransforms.GetBondLength(conf, 0, 1):.3f} Å")
print(f"H–C–C {rdMolTransforms.GetAngleDeg(conf, 2, 0, 1):.1f}°")
print(f"H–C–C–H {rdMolTransforms.GetDihedralDeg(conf, 2, 0, 1, 5):.1f}°")
```

### A torsion scan: butane

```{code-cell} ipython3
butane = to_3d("CCCC")
mp = AllChem.MMFFGetMoleculeProperties(butane)
angles, energies = np.arange(-180, 181, 10), []
for ang in angles:
    m = Chem.Mol(butane)
    ff = AllChem.MMFFGetMoleculeForceField(m, mp)
    ff.MMFFAddTorsionConstraint(0, 1, 2, 3, False, float(ang), float(ang), 1e4)
    ff.Minimize(maxIts=500)
    energies.append(ff.CalcEnergy())
energies = np.array(energies) - min(energies)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(angles, energies, "o-", color="#6c4ad6")
for x, label in [(180, "anti"), (-180, "anti"), (65, "gauche"), (-65, "gauche"), (0, "syn")]:
    ax.annotate(label, (x, np.interp(x, angles, energies)), textcoords="offset points", xytext=(0, 10), ha="center",
                color="#e5487a")
ax.set(xlabel="C–C–C–C dihedral / °", ylabel="Relative MMFF energy / kcal mol$^{-1}$",
       title="Conformational energy of butane")
plt.show()
```

## Conformer ensembles

Flexible molecules exist as many conformers. Generate an ensemble, minimise each, and cluster by RMSD.

```{code-cell} ipython3
ibu = Chem.AddHs(Chem.MolFromSmiles("CC(C)Cc1ccc(cc1)C(C)C(=O)O"))
params = AllChem.ETKDGv3(); params.randomSeed = 7; params.pruneRmsThresh = 0.5
cids = AllChem.EmbedMultipleConfs(ibu, numConfs=50, params=params)
results = AllChem.MMFFOptimizeMoleculeConfs(ibu)
E = np.array([e for _, e in results]); E -= E.min()
print(len(cids), "conformers; lowest five energies:", np.sort(E)[:5].round(2), "kcal/mol")

w = np.exp(-E / 0.593)                           # Boltzmann weights at 298 K (RT = 0.593 kcal/mol)
w /= w.sum()
plt.bar(range(len(E)), np.sort(E), color=plt.cm.viridis(np.sort(E) / E.max()))
plt.xlabel("conformer (sorted)"); plt.ylabel("ΔE / kcal mol$^{-1}$")
plt.title(f"Ibuprofen ensemble: top conformer holds {w.max():.0%} of the population"); plt.show()
```

Overlay the five lowest-energy conformers:

```{code-cell} ipython3
order = np.argsort(E)[:5]
rdMolAlign.AlignMolConformers(ibu, confIds=[int(c) for c in order])
v = py3Dmol.view(width=450, height=330)
colours = ["cyanCarbon", "magentaCarbon", "orangeCarbon", "greenCarbon", "purpleCarbon"]
for i, cid in enumerate(order):
    v.addModel(Chem.MolToMolBlock(ibu, confId=int(cid)), "mol")
    v.setStyle({"model": i}, {"stick": {"colorscheme": colours[i], "radius": 0.12}})
v.zoomTo(); v.show()
```

## 3D shape descriptors

```{code-cell} ipython3
shapes = {"benzene": "c1ccccc1", "hexane": "CCCCCC", "adamantane": "C1C2CC3CC1CC(C2)C3",
          "decane": "CCCCCCCCCC", "naphthalene": "c1ccc2ccccc2c1", "neopentane": "CC(C)(C)C"}
pts = {}
for n, s in shapes.items():
    m = to_3d(s)
    pts[n] = (Descriptors3D.NPR1(m), Descriptors3D.NPR2(m))
fig, ax = plt.subplots(figsize=(5.5, 5))
ax.fill([0, 0.5, 1, 0], [1, 0.5, 1, 1], color="#f4f0ff")
for (x, y), label in [((0, 1), "rod"), ((0.5, 0.5), "disc"), ((1, 1), "sphere")]:
    ax.text(x, y + 0.02, label, ha="center", fontsize=11, weight="bold")
for n, (x, y) in pts.items():
    ax.scatter(x, y, s=80, edgecolor="k"); ax.annotate(n, (x, y), xytext=(5, -12), textcoords="offset points")
ax.set(xlabel="NPR1", ylabel="NPR2", title="Principal-moments-of-inertia shape plot")
plt.show()
```

## Proteins and ligands

py3Dmol can fetch structures from the RCSB Protein Data Bank {cite:p}`berman2000`. The cell below needs an internet connection, so it is shown but not run when the book is built.

```python
v = py3Dmol.view(query="pdb:1HSG", width=600, height=400)     # HIV-1 protease with an inhibitor
v.setStyle({"cartoon": {"color": "spectrum"}})
v.addStyle({"hetflag": True}, {"stick": {"colorscheme": "greenCarbon"}})
v.zoomTo({"hetflag": True})
v.show()
```

## Exercises

**★ 1.** Generate a 3D structure of glucose and display it with a van der Waals surface.

**★★ 2.** Scan the H–C–C–H dihedral of ethane and find the eclipsed–staggered barrier (≈ 3 kcal/mol).

**★★★ 3.** Compare chair and twist-boat cyclohexane energies with MMFF. How many conformers does ETKDG find for cyclohexane, and why?
