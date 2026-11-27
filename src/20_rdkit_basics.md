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

**RDKit** is the open-source toolkit used by pharmaceutical companies and academic labs to work with molecules in code. In this chapter you will draw molecules beautifully, highlight functional groups in colour, and calculate the properties that decide whether a molecule could become a drug.

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

```{code-cell} ipython3
caffeine = Chem.MolFromSmiles("CN1C=NC2=C1C(=O)N(C(=O)N2C)C")
caffeine
```

```{code-cell} ipython3
print(type(caffeine).__name__, caffeine.GetNumAtoms(), "heavy atoms,", caffeine.GetNumBonds(), "bonds")
print("formula:", rdMolDescriptors.CalcMolFormula(caffeine))
```

Invalid SMILES return `None`, so always check:

```{code-cell} ipython3
bad = Chem.MolFromSmiles("C1CC(C")
print(bad is None)
```

## Drawing grids

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

Highlight a functional group by searching for it (Chapter 21 explains the pattern language, SMARTS).

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

```{code-cell} ipython3
serotonin = mols[1]
for atom in serotonin.GetAtoms():
    print(atom.GetIdx(), atom.GetSymbol(), "aromatic" if atom.GetIsAromatic() else "", 
          "H:", atom.GetTotalNumHs(), "hyb:", atom.GetHybridization())
```

```{code-cell} ipython3
ri = serotonin.GetRingInfo()
print("rings:", ri.NumRings(), [len(r) for r in ri.AtomRings()])
for bond in list(serotonin.GetBonds())[:5]:
    print(bond.GetBeginAtom().GetSymbol(), "-", bond.GetEndAtom().GetSymbol(), bond.GetBondType())
```

### Colouring atoms by a property

Gasteiger partial charges, drawn as a colour map:

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

A **descriptor** is a number computed from a structure.

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

RDKit offers more than 200 descriptors in one call:

```{code-cell} ipython3
all_desc = Descriptors.CalcMolDescriptors(caffeine)
print(len(all_desc), "descriptors; a few:", {k: round(v, 2) for k, v in list(all_desc.items())[:6]})
```

## Drug-likeness: Lipinski's rule of five

Lipinski and co-workers observed that most orally absorbed drugs have MW ≤ 500, logP ≤ 5, ≤ 5 H-bond donors and ≤ 10 H-bond acceptors {cite:p}`lipinski1997`.

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

```{code-cell} ipython3
from rdkit.Chem import QED
df["QED"] = [QED.qed(Chem.MolFromSmiles(s)) for s in drugs.values()]
df.sort_values("QED", ascending=False)[["MW", "logP", "QED"]].plot.barh(y="QED", color="#6c4ad6", legend=False, figsize=(6, 4))
plt.xlabel("QED (0 = unlike drugs, 1 = very drug-like)"); plt.show()
```

## Exercises

**★ 1.** Draw the five DNA/RNA bases (adenine, guanine, cytosine, thymine, uracil) in a grid.

**★★ 2.** Highlight every aromatic ring in sildenafil in teal and the sulfonamide in rose.

**★★★ 3.** Compute Lipinski properties for 20 drugs of your choice, and colour a MW–logP plot by therapeutic class.

```{admonition} 🤖 ML corner
:class: ml-corner
Descriptor tables like `df` are the input to classical machine-learning models. In Chapter 36 we use them to predict solubility and toxicity.
```
