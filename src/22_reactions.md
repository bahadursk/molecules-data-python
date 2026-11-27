---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# RDKit III: Chemical Reactions in Code

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 19–21 and organic chemistry.
```

Reactions can be written in code just like molecules. With **reaction SMARTS** we can draw schemes, apply transformations to new substrates, and enumerate thousands of virtual products, the basis of combinatorial chemistry and of ML retrosynthesis.

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdChemReactions, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
import itertools
import pandas as pd
import matplotlib.pyplot as plt
```

## Reaction SMILES and drawing schemes

The format is `reactants>agents>products`.

```{code-cell} ipython3
def draw_reaction(rxn, size=(800, 220)):
    d = rdMolDraw2D.MolDraw2DSVG(*size)
    d.DrawReaction(rxn, highlightByReactant=True)
    d.FinishDrawing()
    return SVG(d.GetDrawingText())

esterification = AllChem.ReactionFromSmarts("CC(=O)O.OCC>[H+]>CC(=O)OCC.O", useSmiles=True)
draw_reaction(esterification)
```

### Atom mapping shows where atoms go

```{code-cell} ipython3
mapped = AllChem.ReactionFromSmarts("[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH3:7]>>[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH3:7].[OH2:4]",
                                    useSmiles=True)
draw_reaction(mapped)
```

The colours follow each reactant's atoms into the product. Isotope-labelling experiments show the alkoxy oxygen of the ester comes from the alcohol, just as the mapping says.

## Running reactions on new substrates

A reaction **template** written in SMARTS can be applied to any molecule that matches it.

```{code-cell} ipython3
amide_coupling = AllChem.ReactionFromSmarts("[C:1](=[O:2])[OH].[N;H2,H1;!$(NC=O):3]>>[C:1](=[O:2])[N:3]")
acid = Chem.MolFromSmiles("OC(=O)c1ccccc1")
amines = ["NCc1ccccc1", "C1CCNCC1", "Nc1ccc(F)cc1", "CN"]
products = [amide_coupling.RunReactants((acid, Chem.MolFromSmiles(a)))[0][0] for a in amines]
for p in products:
    Chem.SanitizeMol(p)
Draw.MolsToGridImage(products, legends=[Chem.MolToSmiles(p) for p in products], molsPerRow=4, subImgSize=(220, 170))
```

## A colourful named-reaction gallery

```{code-cell} ipython3
named = {
    "Diels–Alder": "C=CC=C.C=CC(=O)OC>>COC(=O)C1CCC=CC1",
    "Suzuki coupling": "Brc1ccccc1.OB(O)c1ccc(C)cc1>[Pd]>Cc1ccc(-c2ccccc2)cc1",
    "Grignard addition": "C[Mg]Br.O=Cc1ccccc1>>CC(O)c1ccccc1",
    "Wittig": "O=Cc1ccccc1.C=P(c1ccccc1)(c1ccccc1)c1ccccc1>>C=Cc1ccccc1.O=P(c1ccccc1)(c1ccccc1)c1ccccc1",
    "Friedel–Crafts acylation": "c1ccccc1.CC(=O)Cl>[Al](Cl)(Cl)Cl>CC(=O)c1ccccc1.Cl",
    "Aldol condensation": "CC(=O)C.O=Cc1ccccc1>>CC(=O)/C=C/c1ccccc1.O",
}
for name, smi in named.items():
    print(name)
    display(draw_reaction(AllChem.ReactionFromSmarts(smi, useSmiles=True), size=(820, 180)))
```

## Enumerating a virtual library

A two-component reductive amination with 6 aldehydes and 6 amines gives 36 products:

```{code-cell} ipython3
red_am = AllChem.ReactionFromSmarts("[CH1:1]=O.[N;H2,H1;!$(NC=O):2]>>[CH2:1][N:2]")
aldehydes = {"benzaldehyde": "O=Cc1ccccc1", "4-F-benzaldehyde": "O=Cc1ccc(F)cc1", "furfural": "O=Cc1ccco1",
             "isobutyraldehyde": "CC(C)C=O", "pyridine-3-CHO": "O=Cc1cccnc1", "cinnamaldehyde": "O=C/C=C/c1ccccc1"}
amines = {"morpholine": "C1COCCN1", "piperazine-Me": "CN1CCNCC1", "aniline": "Nc1ccccc1",
          "benzylamine": "NCc1ccccc1", "cyclopropylamine": "NC1CC1", "glycine Me ester": "COC(=O)CN"}
rows = []
for (an, a), (bn, b) in itertools.product(aldehydes.items(), amines.items()):
    p = red_am.RunReactants((Chem.MolFromSmiles(a), Chem.MolFromSmiles(b)))[0][0]
    Chem.SanitizeMol(p)
    rows.append({"aldehyde": an, "amine": bn, "smiles": Chem.MolToSmiles(p), "MW": Descriptors.MolWt(p),
                 "logP": Descriptors.MolLogP(p)})
lib = pd.DataFrame(rows)
print(len(lib), "products")
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in lib.smiles[:12]], molsPerRow=6, subImgSize=(160, 130))
```

```{code-cell} ipython3
pivot = lib.pivot(index="aldehyde", columns="amine", values="logP")
import seaborn as sns
plt.figure(figsize=(8, 4.5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Spectral_r")
plt.title("cLogP across the virtual library"); plt.show()
```

## Retrosynthesis: running templates backwards

A template written product-first disconnects a target into possible starting materials.

```{code-cell} ipython3
retro_templates = {
    "amide disconnection": "[C:1](=[O:2])[N:3]>>[C:1](=[O:2])O.[N:3]",
    "ester disconnection": "[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])O.[O:3][C:4]",
    "Suzuki disconnection": "[c:1]-!@[c:2]>>[c:1]Br.[c:2]B(O)O",
}
target = Chem.MolFromSmiles("CC(=O)Nc1ccc(-c2ccc(C(=O)OC)cc2)cc1")
display(Draw.MolToImage(target, size=(350, 200), legend="target"))
for name, smarts in retro_templates.items():
    rxn = AllChem.ReactionFromSmarts(smarts)
    outcomes = set()
    for prods in rxn.RunReactants((target,)):
        try:
            for p in prods:
                Chem.SanitizeMol(p)
            outcomes.add(".".join(sorted(Chem.MolToSmiles(p) for p in prods)))
        except Exception:
            pass
    for o in outcomes:
        print(f"{name:22s} -> {o}")
```

Real retrosynthesis software combines tens of thousands of such templates, learned from reaction databases, with a search algorithm and a neural network that ranks the options {cite:p}`segler2018`. Template-free models treat reaction prediction as translating reactant SMILES into product SMILES {cite:p}`schwaller2019`. We return to this in Chapter 37.

## Exercises

**★ 1.** Write a reaction SMARTS for the hydrolysis of an ester and apply it to aspirin.

**★★ 2.** Enumerate all amides from four acids and four amines and plot their MW versus TPSA.

**★★★ 3.** Write a forward template for Williamson ether synthesis and a matching retro-template. Check that applying one then the other returns your starting materials.
