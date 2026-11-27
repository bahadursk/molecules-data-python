---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Crystals and Materials: ASE and pymatgen

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 12–13 and basic solid-state chemistry.
```

Materials science deals with periodic solids: metals, oxides, semiconductors, zeolites, MOFs. Two Python libraries dominate: the **Atomic Simulation Environment** (ASE) {cite:p}`larsen2017` and **pymatgen** {cite:p}`ong2013`, the engine of the Materials Project.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from ase.build import bulk, molecule, surface, fcc111, add_adsorbate
from ase.visualize.plot import plot_atoms
from pymatgen.core import Structure, Lattice, Composition, Element
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
```

## Unit cells and lattices

A crystal is a lattice (three vectors **a**, **b**, **c**) plus a basis of atoms in fractional coordinates.

```{code-cell} ipython3
nacl = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
print(nacl.composition.reduced_formula, len(nacl), "atoms in the conventional cell")
print(f"a = {nacl.lattice.a:.2f} Å, V = {nacl.volume:.1f} Å³, density = {nacl.density:.3f} g/cm³")
print("space group:", SpacegroupAnalyzer(nacl).get_space_group_symbol())
```

```{code-cell} ipython3
structures = {
    "NaCl (rock salt)": nacl,
    "CsCl": Structure(Lattice.cubic(4.12), ["Cs", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    "ZnS (zinc blende)": Structure.from_spacegroup("F-43m", Lattice.cubic(5.41), ["Zn", "S"], [[0, 0, 0], [0.25, 0.25, 0.25]]),
    "CaF2 (fluorite)": Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.46), ["Ca", "F"], [[0, 0, 0], [0.25, 0.25, 0.25]]),
    "SrTiO3 (perovskite)": Structure(Lattice.cubic(3.905), ["Sr", "Ti", "O", "O", "O"],
                                     [[0, 0, 0], [0.5, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5]]),
}
for name, s in structures.items():
    nbrs = s.get_neighbors(s[0], 5.0)
    d_min = min(n.nn_distance for n in nbrs)
    cn = sum(n.nn_distance < 1.1 * d_min for n in nbrs)      # first coordination shell
    print(f"{name:22s} {s.composition.reduced_formula:7s} ρ = {s.density:5.2f} g/cm³  "
          f"CN({s[0].specie}) = {cn:2d} at {d_min:.2f} Å")
```

## Visualising crystals

```{code-cell} ipython3
from pymatgen.io.ase import AseAtomsAdaptor
fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
for ax, (name, s) in zip(axes, structures.items()):
    atoms = AseAtomsAdaptor.get_atoms(s) * (2, 2, 2)
    plot_atoms(atoms, ax, rotation="15x,-30y,0z", radii=0.45)
    ax.set_title(name, fontsize=10); ax.axis("off")
plt.tight_layout(); plt.show()
```

```{code-cell} ipython3
import py3Dmol
from pymatgen.io.cif import CifWriter
v = py3Dmol.view(width=420, height=360)
v.addModel(str(CifWriter(structures["SrTiO3 (perovskite)"])), "cif")
v.setStyle({"sphere": {"scale": 0.35}, "stick": {"radius": 0.1}})
v.addUnitCell(); v.replicateUnitCell(2, 2, 2); v.zoomTo(); v.show()
```

## Building materials with ASE

```{code-cell} ipython3
cu = bulk("Cu", "fcc", a=3.61)
si = bulk("Si", "diamond", a=5.43)
slab = fcc111("Pt", size=(4, 4, 3), vacuum=10.0)
add_adsorbate(slab, molecule("CO"), height=1.9, position="ontop")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
plot_atoms(si * (2, 2, 2), axes[0], rotation="10x,-20y,0z"); axes[0].set_title("Si (diamond)")
plot_atoms(slab, axes[1], rotation="-80x,0y,0z"); axes[1].set_title("CO on Pt(111), side view")
plot_atoms(slab, axes[2]); axes[2].set_title("Pt(111) slab, top view")
for a in axes: a.axis("off")
plt.tight_layout(); plt.show()
```

### A quick equation of state

ASE's built-in EMT potential is a fast toy model for metals; we use it to fit an equation of state for copper.

```{code-cell} ipython3
from ase.calculators.emt import EMT
from ase.eos import EquationOfState
volumes, energies = [], []
for a in np.linspace(3.45, 3.80, 12):
    atoms = bulk("Cu", "fcc", a=a)
    atoms.calc = EMT()
    volumes.append(atoms.get_volume()); energies.append(atoms.get_potential_energy())
eos = EquationOfState(volumes, energies)
v0, e0, B = eos.fit()
print(f"a0 = {(4 * v0) ** (1/3):.3f} Å (exp. 3.615), B = {B * 160.2177:.0f} GPa (exp. ≈ 140)")   # eV/Å³ → GPa
eos.plot(); plt.show()
```

## Powder X-ray diffraction

pymatgen simulates the powder pattern from Bragg's law and structure factors.

```{code-cell} ipython3
xrd = XRDCalculator(wavelength="CuKa")
fig, axes = plt.subplots(len(structures), 1, figsize=(9, 9), sharex=True)
for ax, (name, s), c in zip(axes, structures.items(), ["#e5487a", "#f2a93b", "#0f9d8a", "#3b82f6", "#6c4ad6"]):
    pat = xrd.get_pattern(s, two_theta_range=(10, 90))
    ax.vlines(pat.x, 0, pat.y, color=c, lw=2)
    for x, y, hkl in zip(pat.x, pat.y, pat.hkls):
        if y > 25:
            ax.text(x, y + 4, "".join(str(i) for i in hkl[0]["hkl"]), ha="center", fontsize=7)
    ax.set_ylim(0, 125); ax.set_ylabel("I"); ax.text(0.99, 0.75, name, transform=ax.transAxes, ha="right", color=c, weight="bold")
axes[-1].set_xlabel("2θ / ° (Cu Kα)")
plt.tight_layout(); plt.show()
```

## Compositions and periodic properties

```{code-cell} ipython3
for f in ["LiFePO4", "Fe2O3", "CH3NH3PbI3", "YBa2Cu3O7"]:
    c = Composition(f)
    print(f"{f:12s} M = {c.weight:8.2f}  mean Pauling χ = {np.mean([Element(e.symbol).X for e in c.elements]):.2f}  "
          f"fractions = {{{', '.join(f'{e}: {c.get_atomic_fraction(e):.2f}' for e in c.elements)}}}")
```

### Goldschmidt tolerance factor for perovskites

$$t = \frac{r_A + r_X}{\sqrt2\,(r_B + r_X)}$$

Perovskites usually form when 0.8 ≲ t ≲ 1.0.

```{code-cell} ipython3
radii = {"Sr": 1.44, "Ba": 1.61, "Ca": 1.34, "Pb": 1.49, "Cs": 1.88, "MA": 2.17,
         "Ti": 0.605, "Zr": 0.72, "Sn": 0.69, "PbB": 1.19, "O": 1.40, "I": 2.20, "Br": 1.96}
combos = [("Sr", "Ti", "O"), ("Ba", "Ti", "O"), ("Ca", "Ti", "O"), ("Ba", "Zr", "O"), ("Pb", "Ti", "O"),
          ("Cs", "PbB", "I"), ("MA", "PbB", "I"), ("Cs", "PbB", "Br"), ("MA", "Sn", "I")]
t = [(radii[a] + radii[x]) / (np.sqrt(2) * (radii[b] + radii[x])) for a, b, x in combos]
labels = [f"{a}{b.replace('PbB', 'Pb')}{x}{'3'}" for a, b, x in combos]
plt.figure(figsize=(8, 3.8))
plt.bar(labels, t, color=["#0f9d8a" if 0.8 <= v <= 1.0 else "#e5487a" for v in t])
plt.axhspan(0.8, 1.0, color="#0f9d8a", alpha=0.1); plt.ylim(0.6, 1.15)
plt.ylabel("tolerance factor t"); plt.xticks(rotation=30); plt.title("Will it form a perovskite?")
plt.show()
```

## Data from the Materials Project

The Materials Project {cite:p}`jain2013` holds DFT-computed properties for over 150 000 inorganic materials. With a free API key:

```python
from mp_api.client import MPRester
with MPRester("YOUR_API_KEY") as mpr:
    docs = mpr.materials.summary.search(chemsys="Li-Fe-P-O", fields=["formula_pretty", "band_gap", "energy_above_hull"])
    lfp = mpr.get_structure_by_material_id("mp-19017")
```

Chapter 41 uses data of this kind to train models that predict band gaps and stability.

## Exercises

**★ 1.** Build the wurtzite form of ZnO and compare its density with zinc blende ZnS.

**★★ 2.** Simulate the XRD patterns of anatase and rutile TiO₂ and identify one peak that distinguishes them.

**★★★ 3.** Use EMT to compute the surface energy of Cu(111) from slabs of increasing thickness.
