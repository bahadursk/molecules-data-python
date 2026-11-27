---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# RDKit I: Drawing and Describing Molecules

```{admonition} 🟢 Beginner · school & first-year
:class: level-1
Only Chapters 1–6 are needed. A great first taste of cheminformatics.
```

**RDKit** is a free, open-source toolkit for **cheminformatics**, the use of computers to store, search and analyse chemical structures. It is used by pharmaceutical companies and academic labs alike to read molecules from SMILES or files, draw them, compute their properties, search for substructures, compare them and generate 3D structures. Learning RDKit means a molecule stops being a picture and becomes an object you can question in code: how many rings does it have, how polar is it, would it be absorbed as a tablet? In this chapter you will draw molecules beautifully, highlight functional groups in colour, inspect atoms and bonds, and calculate the properties that decide whether a molecule could become a drug.

The imports below bring in RDKit's core module `Chem`, its drawing tools and several property calculators. The two `IPythonConsole` settings make molecules display as crisp SVG pictures at a convenient size.

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Draw, Descriptors, Crippen, rdMolDescriptors, AllChem
from rdkit.Chem.Draw import rdMolDraw2D, IPythonConsole
from IPython.display import SVG
import pandas as pd
import matplotlib.pyplot as plt
IPythonConsole.ipython_useSVG = True
IPythonConsole.molSize = (350, 250)
```

## Molecules from SMILES

Almost every RDKit workflow starts by turning a text representation into a **molecule object** (type `Mol`). `Chem.MolFromSmiles` reads a SMILES string (Chapter 19), works out the implicit hydrogens, checks the valences and perceives aromaticity. In a notebook, a `Mol` on the last line of a cell is displayed as a drawing, which is the quickest way to check that a SMILES means what you think.

```{code-cell} ipython3
Chem.MolFromSmiles("c1ccccc1O")      # phenol
```

```{code-cell} ipython3
caffeine = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
caffeine
```

```{code-cell} ipython3
print(type(caffeine).__name__, caffeine.GetNumAtoms(), "heavy atoms,", caffeine.GetNumBonds(), "bonds")
print("formula:", rdMolDescriptors.CalcMolFormula(caffeine))
```

The molecule object is a store of chemical information that you query with methods such as `GetNumAtoms()`. By default RDKit counts only heavy (non-hydrogen) atoms, because the hydrogens are implicit.

If a SMILES string breaks the grammar or describes an impossible valence (such as a carbon with five bonds), RDKit prints an error and returns `None` instead of a molecule. Any later call on `None` would fail with a confusing message, so always check when reading structures from a file or a user:

```{code-cell} ipython3
bad = Chem.MolFromSmiles("C1CC(C")
print(bad is None)
```

## Drawing grids

Comparing structures side by side is much easier than looking at them one at a time. `Draw.MolsToGridImage` takes a list of molecules and draws them in a grid, with optional `legends` (one label per molecule), `molsPerRow` and the size of each cell in pixels. Here the common neurotransmitters show their shared motif, an amine attached by a short chain to an aromatic ring or an acid.

```{code-cell} ipython3
neurotransmitters = {
    "dopamine": "NCCc1ccc(O)c(O)c1", "serotonin": "NCCc1c[nH]c2ccc(O)cc12",
    "adrenaline": "CNC[C@H](O)c1ccc(O)c(O)c1", "GABA": "NCCCC(=O)O",
    "acetylcholine": "CC(=O)OCC[N+](C)(C)C", "histamine": "NCCc1c[nH]cn1",
    "glutamate": "N[C@@H](CCC(=O)O)C(=O)O", "melatonin": "CC(=O)NCCc1c[nH]c2ccc(OC)cc12",
}
mols = [Chem.MolFromSmiles(s) for s in neurotransmitters.values()]
Draw.MolsToGridImage(mols, legends=list(neurotransmitters), molsPerRow=4, subImgSize=(220, 180))
```

## Colour highlighting

Highlighting parts of a structure in colour draws the eye to what matters: a functional group, a reactive site or the atoms a model thinks are important. The steps are to find the atoms (a substructure search, explained with its pattern language SMARTS in Chapter 21), then pass their indices to the drawing function. The basic version takes a single pattern and needs only two lines:

```{code-cell} ipython3
phenol = Chem.MolFromSmiles("Oc1ccc(cc1)C(=O)O")      # 4-hydroxybenzoic acid
match = phenol.GetSubstructMatch(Chem.MolFromSmarts("C(=O)O"))
print(match)                                          # indices of the carboxylic acid atoms
Draw.MolToImage(phenol, highlightAtoms=match, size=(260, 200))
```

For several groups in several colours, the drawing options must be set by hand. The function below collects, for each pattern, the matching atoms and the bonds between them, assigns each its colour as an (R, G, B, transparency) tuple, and draws the molecule with RDKit's lower-level `MolDraw2DSVG` drawer.

```{code-cell} ipython3
def draw_highlighted(mol, patterns, size=(500, 320), legend=""):
    """Highlight several SMARTS patterns, each in its own colour."""
    atoms, bonds, acol, bcol = [], [], {}, {}
    for smarts, colour in patterns:
        patt = Chem.MolFromSmarts(smarts)
        for match in mol.GetSubstructMatches(patt):
            for a in match:
                atoms.append(a); acol[a] = colour
            for b in patt.GetBonds():
                bond = mol.GetBondBetweenAtoms(match[b.GetBeginAtomIdx()], match[b.GetEndAtomIdx()])
                bonds.append(bond.GetIdx()); bcol[bond.GetIdx()] = colour
    d = rdMolDraw2D.MolDraw2DSVG(*size)
    d.drawOptions().highlightBondWidthMultiplier = 18
    d.drawOptions().legendFontSize = 20
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=atoms, highlightAtomColors=acol,
                                       highlightBonds=bonds, highlightBondColors=bcol, legend=legend)
    d.FinishDrawing()
    return SVG(d.GetDrawingText())

TEAL, ROSE, AMBER, VIOLET = (0.06, 0.62, 0.54, 0.5), (0.9, 0.28, 0.48, 0.5), (0.95, 0.66, 0.23, 0.6), (0.42, 0.29, 0.84, 0.45)

atorvastatin = Chem.MolFromSmiles("CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O")
draw_highlighted(atorvastatin, [("C(=O)[OX2H1]", ROSE), ("C(=O)N", TEAL), ("[CX4][OX2H]", AMBER), ("cF", VIOLET)],
                 legend="atorvastatin: acid (rose), amide (teal), alcohols (amber), C–F (violet)")
```

## Atoms, bonds and rings

A `Mol` contains `Atom` and `Bond` objects (the same design as our own classes in Chapter 10), and each can be asked about itself. `GetAtoms()` loops over the atoms, and methods such as `GetSymbol()`, `GetIsAromatic()`, `GetTotalNumHs()` and `GetHybridization()` describe each one. This is how you count specific atom types, find stereocentres, or build the per-atom features that graph neural networks use (Chapter 34).

```{code-cell} ipython3
serotonin = mols[1]
for atom in serotonin.GetAtoms():
    print(atom.GetIdx(), atom.GetSymbol(), "aromatic" if atom.GetIsAromatic() else "", 
          "H:", atom.GetTotalNumHs(), "hyb:", atom.GetHybridization())
```

Ring information is computed once and stored in a `RingInfo` object: how many rings there are and which atoms form each one. Serotonin's indole gives one five-membered and one six-membered ring. Bonds report the atoms at each end and their bond type; aromatic bonds have their own type rather than being single or double.

```{code-cell} ipython3
ri = serotonin.GetRingInfo()
print("rings:", ri.NumRings(), [len(r) for r in ri.AtomRings()])
for bond in list(serotonin.GetBonds())[:5]:
    print(bond.GetBeginAtom().GetSymbol(), "-", bond.GetEndAtom().GetSymbol(), bond.GetBondType())
```

### Colouring atoms by a property

Instead of highlighting a few atoms, you can colour every atom by a number. **Partial charges** describe how electron density is shared unevenly between bonded atoms: electronegative atoms such as oxygen carry a small negative charge and their neighbours a small positive one. The **Gasteiger** method estimates partial charges quickly from electronegativities, and RDKit stores the result as a property on each atom. `SimilarityMaps.GetSimilarityMapFromWeights` draws a smooth colour map from one number per atom:

```{code-cell} ipython3
from rdkit.Chem.Draw import SimilarityMaps
m = Chem.MolFromSmiles("CC(=O)Nc1ccc(O)cc1")     # paracetamol
AllChem.ComputeGasteigerCharges(m)
charges = [a.GetDoubleProp("_GasteigerCharge") for a in m.GetAtoms()]
d = rdMolDraw2D.MolDraw2DCairo(450, 330)
SimilarityMaps.GetSimilarityMapFromWeights(m, charges, colorMap="coolwarm", contourLines=8, draw2d=d)
d.FinishDrawing()
from IPython.display import Image
Image(d.GetDrawingText())
```

Red regions carry negative partial charge (oxygen, nitrogen); blue regions are positive.

## Molecular descriptors

A **descriptor** is a number computed from a structure that captures some aspect of it: its size, its polarity, its flexibility. Descriptors turn molecules into rows of numbers, which is what makes it possible to sort, filter, plot and, later, model them with machine learning. The most common ones each have a chemical meaning. **MW** is the molecular weight. **logP** is the predicted logarithm of the octanol–water partition coefficient, a measure of how oily (lipophilic) a molecule is. **HBD** and **HBA** count hydrogen-bond donors and acceptors. **TPSA**, the topological polar surface area, is the surface area of the polar atoms and predicts how easily a molecule crosses cell membranes. **RotB** counts rotatable bonds, a measure of flexibility, and **fsp3** is the fraction of carbons that are sp³, a measure of three-dimensionality.

A single descriptor is one function call:

```{code-cell} ipython3
print(Descriptors.MolWt(caffeine), Crippen.MolLogP(caffeine), rdMolDescriptors.CalcTPSA(caffeine))
```

Computing eight descriptors for ten drugs and collecting them in a pandas DataFrame gives a property table:

```{code-cell} ipython3
drugs = {
    "aspirin": "CC(=O)Oc1ccccc1C(=O)O", "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "paracetamol": "CC(=O)Nc1ccc(O)cc1", "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
    "penicillin G": "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O",
    "morphine": "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5",
    "atorvastatin": Chem.MolToSmiles(atorvastatin),
    "cyclosporin": "CC[C@H]1C(=O)N(CC(=O)N([C@H](C(=O)N[C@H](C(=O)N([C@H](C(=O)N[C@H](C(=O)N[C@@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N([C@H](C(=O)N1)[C@@H]([C@H](C)C/C=C/C)O)C)C(C)C)C)CC(C)C)C)CC(C)C)C)C)C)CC(C)C)C)C(C)C)CC(C)C)C)C",
    "metformin": "CN(C)C(=N)NC(=N)N", "sildenafil": "CCCc1nn(C)c2c(=O)[nH]c(-c3cc(S(=O)(=O)N4CCN(C)CC4)ccc3OCC)nc12",
}

def describe(smi):
    m = Chem.MolFromSmiles(smi)
    return {"MW": Descriptors.MolWt(m), "logP": Crippen.MolLogP(m), "HBD": rdMolDescriptors.CalcNumHBD(m),
            "HBA": rdMolDescriptors.CalcNumHBA(m), "TPSA": rdMolDescriptors.CalcTPSA(m),
            "RotB": rdMolDescriptors.CalcNumRotatableBonds(m), "rings": rdMolDescriptors.CalcNumRings(m),
            "fsp3": rdMolDescriptors.CalcFractionCSP3(m)}

df = pd.DataFrame({name: describe(s) for name, s in drugs.items()}).T.round(2)
df
```

RDKit offers more than 200 descriptors in one call, returned as a dictionary from name to value:

```{code-cell} ipython3
all_desc = Descriptors.CalcMolDescriptors(caffeine)
print(len(all_desc), "descriptors; a few:", {k: round(v, 2) for k, v in list(all_desc.items())[:6]})
```

## Drug-likeness: Lipinski's rule of five

A drug taken as a tablet must dissolve in the gut and then pass through cell membranes into the blood. Molecules that are too large, too oily or too polar tend to fail at one of these steps. Lipinski and co-workers studied thousands of drug candidates and observed that most orally absorbed drugs have MW ≤ 500, logP ≤ 5, ≤ 5 H-bond donors and ≤ 10 H-bond acceptors {cite:p}`lipinski1997`. Because every limit is a multiple of five, this is known as the **rule of five**, and a molecule that breaks more than one rule is less likely to be orally active. Counting violations is a one-line calculation on the descriptor table: each comparison gives `True` or `False` per drug, and adding them counts the `True`s.

```{code-cell} ipython3
df["Ro5 violations"] = ((df.MW > 500).astype(int) + (df.logP > 5) + (df.HBD > 5) + (df.HBA > 10))
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.add_patch(plt.Rectangle((0, -2), 500, 7, color="#0f9d8a", alpha=0.12, label="rule-of-five space"))
sc = ax.scatter(df.MW, df.logP, s=df.TPSA * 3, c=df["Ro5 violations"], cmap="plasma", edgecolor="k", vmin=0, vmax=2)
for name, r in df.iterrows():
    ax.annotate(name, (r.MW, r.logP), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.set(xlabel="Molecular weight / g mol$^{-1}$", ylabel="cLogP", title="Drug-likeness map (bubble size = TPSA)")
plt.colorbar(sc, label="Ro5 violations"); ax.legend(loc="upper right")
plt.show()
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Cyclosporin breaks the rules yet is orally active: it hides its polar groups by internal hydrogen bonding. Rules of thumb are guides, not laws.
```

### QED: a single drug-likeness score

Pass/fail rules throw away information: a molecule with MW 501 is treated as harshly as one with MW 900. The **quantitative estimate of drug-likeness** (QED) instead combines eight properties into one smooth score between 0 and 1, based on how typical each value is among known oral drugs. It is useful for ranking candidates rather than filtering them.

```{code-cell} ipython3
from rdkit.Chem import QED
df["QED"] = [QED.qed(Chem.MolFromSmiles(s)) for s in drugs.values()]
df.sort_values("QED", ascending=False)[["MW", "logP", "QED"]].plot.barh(y="QED", color="#6c4ad6", legend=False, figsize=(6, 4))
plt.xlabel("QED (0 = unlike drugs, 1 = very drug-like)"); plt.show()
```

```{admonition} Summary
:class: tip
- `Chem.MolFromSmiles` turns text into a `Mol` object; check for `None` on bad input.
- `Draw.MolsToGridImage` compares structures; `highlightAtoms` and `MolDraw2DSVG` highlight groups in colour.
- Atoms, bonds and rings are objects with query methods (`GetSymbol`, `GetIsAromatic`, `GetRingInfo`, …).
- Descriptors such as MW, logP, HBD, HBA, TPSA and RotB turn molecules into numbers; `CalcMolDescriptors` gives 200+.
- Lipinski's rule of five and QED estimate oral drug-likeness; they are guides, not laws.
```

## Exercises

**★ 1.** Draw the five DNA/RNA bases (adenine, guanine, cytosine, thymine, uracil) in a grid.

**★★ 2.** Highlight every aromatic ring in sildenafil in teal and the sulfonamide in rose.

**★★★ 3.** Compute Lipinski properties for 20 drugs of your choice, and colour a MW–logP plot by therapeutic class.

```{admonition} 🤖 ML corner
:class: ml-corner
Descriptor tables like `df` are the input to classical machine-learning models. In Chapter 36 we use them to predict solubility and toxicity.
```
