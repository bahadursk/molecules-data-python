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

Materials science deals with periodic solids: metals, oxides, semiconductors, zeolites, MOFs. Unlike a molecule, a crystal has no natural end: a small repeating unit is copied in three directions to fill space, and the properties of the material (its density, hardness, colour, conductivity, catalytic activity) follow from what that unit contains and how it repeats. Describing crystals in code therefore needs different tools from RDKit: lattices, fractional coordinates, symmetry, surfaces and diffraction patterns. Two Python libraries dominate: the **Atomic Simulation Environment** (ASE) {cite:p}`larsen2017` and **pymatgen** {cite:p}`ong2013`, the engine of the Materials Project.

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

A crystal is a lattice (three vectors **a**, **b**, **c**) plus a basis of atoms in fractional coordinates. The **lattice** is the repeating framework: three vectors whose lengths ($a$, $b$, $c$) and angles ($\alpha$, $\beta$, $\gamma$) define the **unit cell**, the box that is stacked to build the crystal. The **basis** lists the atoms inside one cell. Their positions are given as **fractional coordinates**, fractions of the cell vectors between 0 and 1, so that (0.5, 0.5, 0.5) is always the cell centre whatever its size. Converting to ordinary (Cartesian) coordinates in ångströms means multiplying by the cell vectors.

pymatgen's `Lattice` and `Structure` classes store exactly these two ingredients. The simplest example is a cubic cell with one atom at the corner, the simple cubic structure of polonium:

```{code-cell} ipython3
po = Structure(Lattice.cubic(3.35), ["Po"], [[0, 0, 0]])
print(po.lattice.abc, po.lattice.angles)
print(po)
```

Most crystals have **symmetry**: rotations, reflections and translations that map the structure onto itself. The 230 possible combinations are the **space groups**, written in Hermann–Mauguin notation such as Fm-3m for rock salt. Given a space group, a few symmetry-distinct atoms generate all the others, so `Structure.from_spacegroup` needs only one Na and one Cl position to build the full eight-atom conventional cell. pymatgen then computes the density from the cell volume and the atomic masses, and `SpacegroupAnalyzer` recovers the symmetry from any structure.

```{code-cell} ipython3
nacl = Structure.from_spacegroup("Fm-3m", Lattice.cubic(5.64), ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]])
print(nacl.composition.reduced_formula, len(nacl), "atoms in the conventional cell")
print(f"a = {nacl.lattice.a:.2f} Å, V = {nacl.volume:.1f} Å³, density = {nacl.density:.3f} g/cm³")
print("space group:", SpacegroupAnalyzer(nacl).get_space_group_symbol())
```

The **coordination number** (CN) of an atom is the number of nearest neighbours around it, and it is the simplest description of a crystal structure: 6 in rock salt (octahedral), 8 in CsCl (cubic), 4 in zinc blende (tetrahedral). `get_neighbors(site, r)` lists every atom within a radius, wrapping across cell boundaries automatically; counting those within 10 % of the shortest distance gives the first coordination shell. Comparing five classic structure types:

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

A single unit cell rarely shows the pattern clearly, so crystals are usually drawn as a small block of cells, a **supercell**. Converting a pymatgen structure into an ASE `Atoms` object and multiplying it by `(2, 2, 2)` repeats it twice in each direction; ASE's `plot_atoms` draws the result with Matplotlib.

```{code-cell} ipython3
from pymatgen.io.ase import AseAtomsAdaptor
fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
for ax, (name, s) in zip(axes, structures.items()):
    atoms = AseAtomsAdaptor.get_atoms(s) * (2, 2, 2)
    plot_atoms(atoms, ax, rotation="15x,-30y,0z", radii=0.45)
    ax.set_title(name, fontsize=10); ax.axis("off")
plt.tight_layout(); plt.show()
```

For an interactive view, the structure is written in **CIF** (Crystallographic Information File) format, the standard file format of crystallography, and passed to py3Dmol (Chapter 23), which can draw the unit cell and replicate it.

```{code-cell} ipython3
import py3Dmol
from pymatgen.io.cif import CifWriter
v = py3Dmol.view(width=420, height=360)
v.addModel(str(CifWriter(structures["SrTiO3 (perovskite)"])), "cif")
v.setStyle({"sphere": {"scale": 0.35}, "stick": {"radius": 0.1}})
v.addUnitCell(); v.replicateUnitCell(2, 2, 2); v.zoomTo(); v.show()
```

## Building materials with ASE

ASE specialises in building and simulating atomistic systems. `bulk` creates common crystals by name and structure type, `molecule` creates small molecules from a built-in library, and `fcc111` creates a **slab**, a thin layer of a crystal cut along a chosen plane with empty space (vacuum) above and below. Slabs model surfaces, where catalysis happens. `add_adsorbate` places a molecule on a surface site; here CO sits on top of a platinum atom, the starting point of every computational catalysis study (Chapter 38). The basic object, `Atoms`, holds symbols, positions and the cell:

```{code-cell} ipython3
print(bulk("Cu", "fcc", a=3.61))
print(molecule("H2O").get_chemical_symbols(), molecule("H2O").positions.round(3).tolist())
```

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

An **equation of state** relates a solid's energy to its volume. Squeezing a crystal or stretching it raises its energy; the minimum gives the equilibrium lattice constant, and the curvature at the minimum gives the **bulk modulus** $B$, which measures how hard the solid is to compress. In ASE, attaching a **calculator** to an `Atoms` object lets you ask for its energy and forces; the calculator might be a full DFT code or, as here, a cheap model. ASE's built-in EMT potential is a fast toy model for metals; we use it to fit an equation of state for copper. `EquationOfState` fits a standard curve through the energies and returns the minimum volume, energy and bulk modulus (in eV/Å³, converted to GPa).

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

X-rays scattered by the regularly spaced planes of atoms in a crystal reinforce each other only at particular angles, given by **Bragg's law**, $n\lambda = 2d\sin\theta$, where $d$ is the spacing between planes. Each set of planes is labelled by its **Miller indices** (hkl). A powder sample contains crystallites in every orientation, so all the allowed reflections appear at once as peaks at angles 2θ; the peak positions come from the cell size and shape, and their intensities from which atoms sit where (the **structure factor**). The pattern is a fingerprint used to identify phases. Bragg's law alone gives the peak positions; for the (200) planes of rock salt with Cu Kα radiation (1.5406 Å):

```{code-cell} ipython3
d_200 = 5.64 / 2                                     # spacing of (200) planes in a cubic cell, Å
print(f"2θ = {2 * np.degrees(np.arcsin(1.5406 / (2 * d_200))):.2f}°")
```

pymatgen simulates the powder pattern from Bragg's law and structure factors. Reflections with intensity above 25 % are labelled with their hkl.

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

Before any structure is known, a material's **composition** already carries a lot of information. pymatgen's `Composition` parses a formula (including groups such as CH₃NH₃), computes its weight and atomic fractions, and links each element to an `Element` object with tabulated properties such as the Pauling electronegativity `X`, atomic radius and oxidation states. Statistics of such properties over the elements in a formula are the basis of composition-only machine-learning models (Chapters 32 and 41).

```{code-cell} ipython3
for f in ["LiFePO4", "Fe2O3", "CH3NH3PbI3", "YBa2Cu3O7"]:
    c = Composition(f)
    print(f"{f:12s} M = {c.weight:8.2f}  mean Pauling χ = {np.mean([Element(e.symbol).X for e in c.elements]):.2f}  "
          f"fractions = {{{', '.join(f'{e}: {c.get_atomic_fraction(e):.2f}' for e in c.elements)}}}")
```

### Goldschmidt tolerance factor for perovskites

The **perovskite** structure, ABX₃, has a large A cation in a cage of corner-sharing BX₆ octahedra. It is one of the most important structure types in materials science, found in ferroelectrics, superconductors and solar-cell absorbers. Whether a given combination of ions can adopt it is estimated by a simple geometric rule: the A–X distance must match the B–X distance times $\sqrt{2}$, as it does in the ideal cubic cell. The **Goldschmidt tolerance factor** measures the mismatch from ionic radii:

$$t = \frac{r_A + r_X}{\sqrt2\,(r_B + r_X)}$$

Perovskites usually form when 0.8 ≲ t ≲ 1.0. Below that range the A ion is too small and other structures (such as ilmenite) compete; above it, A is too large and hexagonal structures form. "MA" stands for the methylammonium cation, CH₃NH₃⁺, of the hybrid perovskites used in solar cells, and "PbB" marks Pb²⁺ on the B site (its radius differs from Pb²⁺ on the A site).

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

The Materials Project {cite:p}`jain2013` holds DFT-computed properties for over 150 000 inorganic materials: structures, formation energies, band gaps, elastic constants and more, each with an identifier such as `mp-19017`. Because pymatgen is the project's own library, downloaded structures arrive as `Structure` objects ready for everything in this chapter. With a free API key:

```python
from mp_api.client import MPRester
with MPRester("YOUR_API_KEY") as mpr:
    docs = mpr.materials.summary.search(chemsys="Li-Fe-P-O", fields=["formula_pretty", "band_gap", "energy_above_hull"])
    lfp = mpr.get_structure_by_material_id("mp-19017")
```

Chapter 41 uses data of this kind to train models that predict band gaps and stability.

```{admonition} Summary
:class: tip
- A crystal = lattice (cell vectors) + basis (atoms in fractional coordinates); space groups describe its symmetry.
- pymatgen `Structure` gives density, symmetry, neighbours and coordination numbers; ASE builds bulk crystals, molecules, slabs and adsorbates.
- Supercells and CIF files help visualise crystals.
- An equation of state gives the lattice constant and bulk modulus.
- Powder XRD peaks follow Bragg's law, $n\lambda = 2d\sin\theta$; intensities come from structure factors.
- `Composition` and `Element` give composition-level properties; the tolerance factor screens perovskites.
```

## Exercises

**★ 1.** Build the wurtzite form of ZnO and compare its density with zinc blende ZnS.

**★★ 2.** Simulate the XRD patterns of anatase and rutile TiO₂ and identify one peak that distinguishes them.

**★★★ 3.** Use EMT to compute the surface energy of Cu(111) from slabs of increasing thickness.
