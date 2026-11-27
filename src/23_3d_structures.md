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

SMILES and fingerprints treat a molecule as a flat graph, but real molecules are three-dimensional and flexible. Their shape decides how they pack in crystals, how they bind to a protein's pocket, how they smell, and what their dipole moment is. Single bonds rotate, so most molecules exist as a mixture of shapes called **conformers**, and the lowest-energy ones dominate. Getting 3D coordinates from a SMILES string is therefore the first step of docking, quantum chemistry (Chapter 28), molecular dynamics (Chapter 29) and 3D machine-learning models. In this chapter we generate 3D coordinates, optimise them with force fields, measure geometry, explore conformers and their energies, and view everything interactively with **py3Dmol**, a molecular viewer that runs in the notebook.

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms, rdMolAlign, Descriptors3D
import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
```

## From SMILES to 3D

Turning a graph into coordinates is called **embedding**. RDKit does it in three steps. First the hydrogens are made explicit with `Chem.AddHs`, because their positions matter in 3D. Then `EmbedMolecule` generates a starting geometry by **distance geometry**: it builds a table of allowed ranges for every interatomic distance from bond lengths and angles and finds coordinates consistent with them, using a random start (hence the seed). Finally a **force field** relaxes the geometry. A force field is a simple classical model that gives the energy of a molecule as a sum of terms for bond stretching, angle bending, torsions and non-bonded interactions; minimising that energy moves the atoms to a sensible structure. The most basic case is methane, whose tetrahedral angle comes out automatically:

```{code-cell} ipython3
methane = Chem.AddHs(Chem.MolFromSmiles("C"))
AllChem.EmbedMolecule(methane, randomSeed=1)
AllChem.MMFFOptimizeMolecule(methane)
conf = methane.GetConformer()
print(conf.GetAtomPosition(1).x, conf.GetAtomPosition(1).y, conf.GetAtomPosition(1).z)
print(f"H–C–H angle {rdMolTransforms.GetAngleDeg(conf, 1, 0, 2):.1f}°")
```

The coordinates are stored in a **conformer** object attached to the molecule. Wrapping the three steps in a function makes 3D structures a one-liner:

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

**ETKDG** (Experimental-Torsion Knowledge Distance Geometry) improves on plain distance geometry by adding torsion-angle preferences learned from thousands of crystal structures, so the starting shapes already look realistic. **MMFF94** (the Merck Molecular Force Field) then relaxes the geometry; it was parameterised for drug-like organic molecules and is a good default.

## Interactive 3D viewing

A list of coordinates is hard to picture. **py3Dmol** draws molecules in 3D directly in the notebook (and on the book's web pages): drag to rotate, scroll to zoom. You create a `view`, add a model from a molfile string (`Chem.MolToMolBlock` gives one, coordinates included), choose a style such as `stick`, `sphere` or a combination, and call `zoomTo()` to centre it. A **van der Waals surface** shows the space the molecule occupies, which is what a protein "sees".

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

A `viewergrid` places several independent viewers in one output, which is handy for comparing shapes. Here are the rigid, spherical C₆₀ fullerene and a floppy 18-membered ring, coloured with different carbon colour schemes.

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

Three numbers describe local geometry. A **bond length** is the distance between two bonded atoms. A **bond angle** is the angle between two bonds at a shared atom. A **dihedral** (torsion) angle involves four atoms A–B–C–D and measures the twist around the B–C bond: the angle between the plane of A, B, C and the plane of B, C, D. Dihedrals are what change between conformers. The `rdMolTransforms` module measures (and can set) all three from a conformer, given atom indices. For ethane, the staggered conformation has H–C–C–H dihedrals near ±60° and 180°.

```{code-cell} ipython3
ethane = to_3d("CC")
conf = ethane.GetConformer()
print(f"C–C  {rdMolTransforms.GetBondLength(conf, 0, 1):.3f} Å")
print(f"H–C–C {rdMolTransforms.GetAngleDeg(conf, 2, 0, 1):.1f}°")
print(f"H–C–C–H {rdMolTransforms.GetDihedralDeg(conf, 2, 0, 1, 5):.1f}°")
```

### A torsion scan: butane

A **torsion scan** maps how the energy changes as one dihedral is rotated. At each angle the dihedral is held fixed by a strong constraint while the rest of the molecule relaxes, and the energy is recorded. For butane, rotating the central C–C bond gives the classic textbook curve: the *anti* conformer (180°) is lowest, the two *gauche* conformers (about ±65°) lie slightly higher because the methyl groups are closer, and the eclipsed *syn* form (0°) is the highest barrier. `Chem.Mol(butane)` makes a fresh copy at each angle so that the scans do not interfere.

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

A flexible molecule such as ibuprofen has several rotatable bonds, so it can adopt dozens of low-energy shapes. To describe it properly you need an **ensemble**: many conformers, each minimised, with their energies. `EmbedMultipleConfs` generates many starting structures from different random seeds, and `pruneRmsThresh` discards any that are nearly identical to one already kept, measured by the **RMSD** (root-mean-square deviation of atom positions after superposition, in Å). `MMFFOptimizeMoleculeConfs` minimises them all and returns their energies.

How much of the time does the molecule spend in each conformer? At equilibrium the population of each follows the **Boltzmann distribution**, $p_i \propto e^{-\Delta E_i/RT}$, so a conformer 1.4 kcal mol⁻¹ above the minimum is about ten times less populated at room temperature. Normalising the weights so they sum to one gives the populations.

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

Overlaying the five lowest-energy conformers, after aligning them on each other, shows which parts of the molecule are rigid (the benzene ring) and which flop around (the isobutyl and acid groups):

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

Descriptors computed from 3D coordinates capture overall shape, which 2D descriptors cannot. The **principal moments of inertia** describe how a molecule's mass is distributed along three perpendicular axes. Dividing the two smaller moments by the largest gives the **normalised PMI ratios** NPR1 and NPR2, which place every molecule in a triangle whose corners are a perfect rod (such as a long alkyne), a flat disc (benzene) and a sphere (adamantane). Drug discovery uses this plot to check whether a library is too "flat", since many successful drugs are more three-dimensional.

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

Large biomolecules are stored in the **Protein Data Bank** (PDB), where each experimentally determined structure has a four-character code. Proteins are usually drawn as a *cartoon* that shows helices and sheets rather than every atom, with a bound drug (a *ligand*, marked as a "hetero" group) drawn as sticks so that its contacts with the protein are visible. py3Dmol can fetch structures from the RCSB Protein Data Bank {cite:p}`berman2000`. The cell below needs an internet connection, so it is shown but not run when the book is built.

```python
v = py3Dmol.view(query="pdb:1HSG", width=600, height=400)     # HIV-1 protease with an inhibitor
v.setStyle({"cartoon": {"color": "spectrum"}})
v.addStyle({"hetflag": True}, {"stick": {"colorscheme": "greenCarbon"}})
v.zoomTo({"hetflag": True})
v.show()
```

```{admonition} Summary
:class: tip
- 3D structure from SMILES: `AddHs`, embed with ETKDG (seeded), relax with MMFF94.
- py3Dmol views molecules interactively; styles include sticks, spheres and surfaces.
- `rdMolTransforms` measures bond lengths, angles and dihedrals; constrained minimisations give torsion scans.
- Flexible molecules need conformer ensembles; Boltzmann weights $e^{-\Delta E/RT}$ give populations.
- PMI ratios (NPR1, NPR2) classify shape as rod, disc or sphere.
```

## Exercises

**★ 1.** Generate a 3D structure of glucose and display it with a van der Waals surface.

**★★ 2.** Scan the H–C–C–H dihedral of ethane and find the eclipsed–staggered barrier (≈ 3 kcal/mol).

**★★★ 3.** Compare chair and twist-boat cyclohexane energies with MMFF. How many conformers does ETKDG find for cyclohexane, and why?
