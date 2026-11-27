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

So far we have kept data in containers (Chapter 4) and written separate functions to act on it (Chapter 6). That works, but as programs grow, it becomes hard to keep track of which functions belong with which data. **Object-oriented programming** solves this by bundling the two together. A **class** is a blueprint that describes a kind of thing: what data each one holds (its **attributes**) and what it can do (its **methods**, functions that belong to the class). An **object** or **instance** is one particular thing built from that blueprint, as carbon is one particular element. Every library you will use, from NumPy arrays to RDKit molecules and pandas tables, is built from classes, and whenever you have written `text.upper()` you have already called a method on an object. In this chapter we design our own `Element`, `Atom`, `Bond` and `Molecule` classes and, in doing so, learn how RDKit thinks about molecules.

## A first class

A class is defined with the keyword `class`, followed by its name (by convention in CapitalisedWords) and a colon. Inside, methods are written like ordinary functions with one difference: their first parameter is always `self`, which refers to the particular object the method is working on. The special method `__init__` (with two underscores on each side) runs automatically whenever a new instance is created and sets up its attributes by assigning to `self.something`. You create an instance by calling the class like a function, and you reach its attributes and methods with a dot.

The smallest useful class just stores data:

```{code-cell} ipython3
class Element:
    def __init__(self, symbol, Z, mass):
        self.symbol = symbol      # attributes are stored on self
        self.Z = Z
        self.mass = mass

carbon = Element("C", 6, 12.011)
oxygen = Element("O", 8, 15.999)
print(carbon.symbol, carbon.Z, carbon.mass)
print(oxygen.symbol, oxygen.Z)
```

`carbon` and `oxygen` are two separate instances of the same class; each has its own `symbol`, `Z` and `mass`. Compared with a dictionary such as `{"symbol": "C", "Z": 6}`, the class guarantees that every element has the same attributes and lets us attach behaviour. Adding a method gives each element the ability to answer a question about itself, here the number of neutrons in an isotope:

```{code-cell} ipython3
class Element:
    def __init__(self, symbol, Z, mass):
        self.symbol = symbol
        self.Z = Z
        self.mass = mass

    def neutrons(self, A=None):
        """Neutrons in the isotope with mass number A (default: nearest integer mass)."""
        A = A or round(self.mass)
        return A - self.Z

carbon = Element("C", 6, 12.011)
print(carbon.neutrons(), carbon.neutrons(14))
```

When you write `carbon.neutrons(14)`, Python passes `carbon` itself as `self` and 14 as `A`, so the method can read `self.Z`. That is the whole trick behind `self`.

## Special methods

Methods whose names start and end with double underscores, often called *dunder* methods, are hooks that let your objects work with Python's built-in functions and operators. `__repr__` controls what is shown when you print or inspect an object, which matters because the default (`<__main__.Element object at 0x7f…>`) tells you nothing. `__lt__` ("less than") defines what `<` means, which is all `sorted()` needs. `__eq__` defines `==`, and `__hash__` lets objects be stored in sets and used as dictionary keys.

The difference `__repr__` makes is immediate:

```{code-cell} ipython3
print(carbon)            # the default, uninformative representation
```

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
print(PT["C"])
print(sorted([PT["Cl"], PT["H"], PT["O"]]))       # sorted by atomic number
print(PT["O"] == Element("O", 8, 15.999))          # equal because Z is equal
```

`PT` is a small periodic table: a dictionary from symbol to `Element` object, built with a dictionary comprehension (Chapter 7). Defining equality by atomic number matches the chemistry, since an element *is* defined by its Z.

## Atoms, bonds and molecules

A molecule is more than a list of elements. Each **atom** is one occurrence of an element, with its own position and charge; a **bond** joins two atoms and has an order; and a **molecule** owns its atoms and bonds and can compute properties from them. Modelling this with three classes mirrors the way chemists think, and it is the same design that RDKit uses.

Writing `__init__` and `__repr__` for every small data class is repetitive, so Python provides **dataclasses**. Putting `@dataclass` above a class (a *decorator*, which modifies the class) and listing its attributes with type hints is enough: Python writes `__init__`, `__repr__` and `__eq__` for you.

```{code-cell} ipython3
import math
from dataclasses import dataclass, field

@dataclass
class Bond:
    a: int
    b: int
    order: int = 1

print(Bond(0, 1, 2))
print(Bond(0, 2))           # order takes its default
```

Two more tools appear in the `Molecule` class below. `field(default_factory=list)` gives each new molecule its *own* empty list; writing `atoms: list = []` would make every molecule share one list, the copying trap of Chapter 4. And `@property` turns a method into something you read like an attribute, so `mol.formula` is computed fresh from the atoms each time you ask for it and can never get out of date. The formula is written in the **Hill system**, carbon first, hydrogen second, then the other elements alphabetically.

```{code-cell} ipython3
@dataclass
class Atom:
    element: Element
    xyz: tuple = (0.0, 0.0, 0.0)
    charge: int = 0

    def distance(self, other):
        return math.dist(self.xyz, other.xyz)

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

`add_atom` returns the new atom's index, so you can keep it in a variable and use it to make bonds. Building formaldehyde, H₂C=O, with real coordinates in ångströms shows how naturally the classes read:

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

`len(f)` works because we defined `__len__`, and `print(f)` uses our `__repr__`. The object now behaves like a built-in part of Python.

## Inheritance

**Inheritance** lets you define a new class as a specialised version of an existing one. You write the parent class in parentheses, `class Ion(Molecule):`, and the new class (the *subclass*) automatically gets every attribute and method of the parent. You then add new behaviour or replace (*override*) methods that should work differently. This avoids copying code: an ion is a molecule with an overall charge, so it should reuse everything a molecule already knows and add only the charge.

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
print(isinstance(nh4, Molecule))     # an Ion is still a Molecule
```

`add_atom`, `add_bond`, `formula` and `mass` all came from `Molecule` without a single extra line. The loop variable `_` is the conventional name for a value you do not need.

## From our classes to RDKit

RDKit's design is the same idea at industrial strength. A `Mol` object holds `Atom` and `Bond` objects, and each has methods such as `GetSymbol()`, `GetIdx()` and `GetNeighbors()`. Because our classes store the same information, converting one of our molecules into an RDKit molecule is a direct translation: create an editable `RWMol`, add an atom for each of ours, add each bond with the matching bond type, and *sanitise* the result, which makes RDKit check valences and work out implicit hydrogens and aromaticity.

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
```

Walking over an RDKit molecule's atoms uses exactly the methods you would expect from our own design. Here acetic acid is shown with explicit hydrogens:

```{code-cell} ipython3
for atom in Chem.AddHs(Chem.MolFromSmiles("CC(=O)O")).GetAtoms():
    print(atom.GetIdx(), atom.GetSymbol(), [n.GetSymbol() for n in atom.GetNeighbors()])
```

```{code-cell} ipython3
from rdkit.Chem import Draw
Draw.MolToImage(Chem.MolFromSmiles("CC(=O)O"), size=(250, 180))
```

```{admonition} Summary
:class: tip
- A **class** is a blueprint; an **instance** is one object built from it. Attributes hold data, methods hold behaviour.
- `__init__` sets up a new instance; `self` is the instance a method is working on.
- Dunder methods such as `__repr__`, `__lt__`, `__eq__` and `__len__` make objects work with `print`, `sorted`, `==` and `len`.
- `@dataclass` writes the boilerplate for data-holding classes; use `field(default_factory=list)` for list attributes.
- `@property` makes a computed value read like an attribute.
- A subclass inherits everything from its parent and adds or overrides what differs.
- RDKit's `Mol`, `Atom` and `Bond` follow the same design.
```

## Exercises

**★ 1.** Add a `__str__` method to `Element` that returns `"C (6)"`.

```{admonition} Solution
:class: dropdown
`def __str__(self): return f"{self.symbol} ({self.Z})"`. `print()` uses `__str__` when it exists and falls back to `__repr__` otherwise.
```

**★★ 2.** Add a method `Molecule.degree_of_unsaturation()` using $DoU = C - H/2 + N/2 + 1$ (halogens count as H). Test it on benzene.

**★★★ 3.** Write `Molecule.from_xyz(path)` as a `@classmethod` that reads an XYZ file and guesses bonds from distances (bond if distance < 1.2 × sum of covalent radii).
