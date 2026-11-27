---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Object-Oriented Chemistry: Atoms and Molecules as Classes

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 1–9.
```

Every library you will use, from NumPy arrays to RDKit molecules, is built from **classes**. A class bundles data (**attributes**) with the functions that act on it (**methods**). In this chapter we design our own `Atom`, `Bond` and `Molecule` classes and, in doing so, learn how RDKit thinks.

## A first class

```{code-cell} ipython3
class Element:
    def __init__(self, symbol, Z, mass):
        self.symbol = symbol      # attributes are stored on self
        self.Z = Z
        self.mass = mass

    def neutrons(self, A=None):
        """Neutrons in the isotope with mass number A (default: nearest integer mass)."""
        A = A or round(self.mass)
        return A - self.Z

carbon = Element("C", 6, 12.011)
print(carbon.symbol, carbon.Z, carbon.neutrons(), carbon.neutrons(14))
```

`__init__` runs when you create an **instance**. `self` is the instance itself.

## Special methods

Methods with double underscores let your objects work with Python's built-in operators and functions.

```{code-cell} ipython3
class Element:
    def __init__(self, symbol, Z, mass):
        self.symbol, self.Z, self.mass = symbol, Z, mass
    def __repr__(self):
        return f"Element({self.symbol!r}, Z={self.Z})"
    def __lt__(self, other):          # allows sorting
        return self.Z < other.Z
    def __eq__(self, other):
        return isinstance(other, Element) and self.Z == other.Z
    def __hash__(self):
        return hash(self.Z)

PT = {s: Element(s, z, m) for s, z, m in [("H", 1, 1.008), ("C", 6, 12.011), ("N", 7, 14.007),
                                          ("O", 8, 15.999), ("S", 16, 32.06), ("Cl", 17, 35.45)]}
print(sorted([PT["Cl"], PT["H"], PT["O"]]))
```

## Atoms, bonds and molecules

```{code-cell} ipython3
import math
from dataclasses import dataclass, field

@dataclass
class Atom:
    element: Element
    xyz: tuple = (0.0, 0.0, 0.0)
    charge: int = 0

    def distance(self, other):
        return math.dist(self.xyz, other.xyz)

@dataclass
class Bond:
    a: int
    b: int
    order: int = 1

@dataclass
class Molecule:
    name: str
    atoms: list = field(default_factory=list)
    bonds: list = field(default_factory=list)

    def add_atom(self, symbol, xyz=(0, 0, 0), charge=0):
        self.atoms.append(Atom(PT[symbol], xyz, charge))
        return len(self.atoms) - 1

    def add_bond(self, i, j, order=1):
        self.bonds.append(Bond(i, j, order))

    @property
    def formula(self):
        from collections import Counter
        c = Counter(a.element.symbol for a in self.atoms)
        order = ["C", "H"] + sorted(k for k in c if k not in "CH")   # Hill system
        return "".join(f"{el}{c[el] if c[el] > 1 else ''}" for el in order if el in c)

    @property
    def mass(self):
        return sum(a.element.mass for a in self.atoms)

    def neighbours(self, i):
        return [b.b if b.a == i else b.a for b in self.bonds if i in (b.a, b.b)]

    def __len__(self):
        return len(self.atoms)

    def __repr__(self):
        return f"<Molecule {self.name}: {self.formula}, {self.mass:.2f} g/mol>"
```

Build formaldehyde, H₂C=O, with real coordinates:

```{code-cell} ipython3
f = Molecule("formaldehyde")
C = f.add_atom("C", (0.000, 0.000, 0.000))
O = f.add_atom("O", (0.000, 0.000, 1.205))
H1 = f.add_atom("H", (0.000, 0.943, -0.587))
H2 = f.add_atom("H", (0.000, -0.943, -0.587))
f.add_bond(C, O, 2); f.add_bond(C, H1); f.add_bond(C, H2)

print(f, len(f), "atoms")
print("C=O length:", round(f.atoms[C].distance(f.atoms[O]), 3), "Å")
print("neighbours of C:", [f.atoms[i].element.symbol for i in f.neighbours(C)])
```

## Inheritance

A subclass reuses and extends its parent.

```{code-cell} ipython3
class Ion(Molecule):
    @property
    def charge(self):
        return sum(a.charge for a in self.atoms)

    def __repr__(self):
        sign = "+" if self.charge > 0 else "-"
        return f"<Ion {self.formula}{abs(self.charge) if abs(self.charge) > 1 else ''}{sign}>"

nh4 = Ion("ammonium")
n = nh4.add_atom("N", charge=+1)
for _ in range(4):
    nh4.add_bond(n, nh4.add_atom("H"))
print(nh4, f"{nh4.mass:.3f} g/mol")
```

## From our classes to RDKit

RDKit's design is the same idea, industrial strength: a `Mol` holds `Atom` and `Bond` objects with methods such as `GetSymbol()` and `GetNeighbors()`.

```{code-cell} ipython3
from rdkit import Chem

def to_rdkit(mol: Molecule):
    rw = Chem.RWMol()
    for a in mol.atoms:
        at = Chem.Atom(a.element.Z)
        at.SetFormalCharge(a.charge)
        rw.AddAtom(at)
    order = {1: Chem.BondType.SINGLE, 2: Chem.BondType.DOUBLE, 3: Chem.BondType.TRIPLE}
    for b in mol.bonds:
        rw.AddBond(b.a, b.b, order[b.order])
    m = rw.GetMol()
    Chem.SanitizeMol(m)
    return m

rd = to_rdkit(f)
print(Chem.MolToSmiles(rd))
for atom in Chem.AddHs(Chem.MolFromSmiles("CC(=O)O")).GetAtoms():
    print(atom.GetIdx(), atom.GetSymbol(), [n.GetSymbol() for n in atom.GetNeighbors()])
```

```{code-cell} ipython3
from rdkit.Chem import Draw
Draw.MolToImage(Chem.MolFromSmiles("CC(=O)O"), size=(250, 180))
```

## Exercises

**★ 1.** Add a `__str__` method to `Element` that returns `"C (6)"`.

**★★ 2.** Add a method `Molecule.degree_of_unsaturation()` using $DoU = C - H/2 + N/2 + 1$ (halogens count as H). Test it on benzene.

**★★★ 3.** Write `Molecule.from_xyz(path)` as a `@classmethod` that reads an XYZ file and guesses bonds from distances (bond if distance < 1.2 × sum of covalent radii).
