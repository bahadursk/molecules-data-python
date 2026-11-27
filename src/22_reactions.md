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

A reaction is a transformation: some bonds break, others form, and the atoms rearrange into products. Just as SMILES writes a molecule as text, **reaction SMILES** writes a reaction as text, and **reaction SMARTS** writes a *general rule* for a reaction type, such as "any carboxylic acid plus any amine gives an amide". Having reactions in code is useful in three ways. You can draw clean schemes for reports. You can apply a known transformation to new substrates to predict their products, and do it for thousands of combinations to build a **virtual library** before synthesising anything, which is the basis of combinatorial chemistry. And you can run a rule backwards to suggest how a target could be made, which is the core idea behind computer-aided retrosynthesis. This chapter covers each in turn.

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

A reaction SMILES has three parts separated by `>`: `reactants>agents>products`. Molecules within a part are separated by dots, and the agents part (catalysts, solvents, reagents that are not incorporated) may be empty, giving `reactants>>products`. `AllChem.ReactionFromSmarts(text, useSmiles=True)` reads it into a reaction object, which can report its reactants and products:

```{code-cell} ipython3
rxn = AllChem.ReactionFromSmarts("CC(=O)O.OCC>>CC(=O)OCC.O", useSmiles=True)
print(rxn.GetNumReactantTemplates(), "reactants,", rxn.GetNumProductTemplates(), "products")
print([Chem.MolToSmiles(m) for m in rxn.GetReactants()])
```

RDKit can draw a reaction object as a scheme. The helper below wraps the SVG drawer; `highlightByReactant=True` colours each reactant's atoms differently. Here is the Fischer esterification of acetic acid with ethanol, with acid as the agent above the arrow:

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

A reaction SMILES as written says which molecules go in and come out, but not which atom becomes which. **Atom mapping** adds that information: each reactant atom gets a number after a colon (`[C:2]`), and the same number on a product atom says it is the same atom. Mapping is what turns a reaction from a before-and-after picture into a mechanism-level statement, and it is how reaction rules know which bonds change. In the esterification below, map number 4 is the acid's hydroxyl oxygen, which ends up in water, and number 5 is the alcohol's oxygen, which ends up in the ester.

```{code-cell} ipython3
mapped = AllChem.ReactionFromSmarts("[CH3:1][C:2](=[O:3])[OH:4].[OH:5][CH2:6][CH3:7]>>[CH3:1][C:2](=[O:3])[O:5][CH2:6][CH3:7].[OH2:4]",
                                    useSmiles=True)
draw_reaction(mapped)
```

The colours follow each reactant's atoms into the product. Isotope-labelling experiments show the alkoxy oxygen of the ester comes from the alcohol, just as the mapping says.

## Running reactions on new substrates

A reaction **template** is a reaction SMARTS that describes only the atoms that change and their immediate surroundings, using the pattern language of Chapter 21, with mapped atoms carried into the product. Because it is a pattern, it can be applied to *any* molecule that matches it. `rxn.RunReactants((mol1, mol2))` tries every way of matching the reactant patterns and returns a tuple of possible product sets; each set is a tuple of molecules, which is why the code below picks `[0][0]`, the first product of the first outcome. Products come back unsanitised, so `Chem.SanitizeMol` is called to tidy valences and aromaticity before use.

The most basic template turns any primary alcohol into its aldehyde (an oxidation):

```{code-cell} ipython3
oxidation = AllChem.ReactionFromSmarts("[CH2:1][OH:2]>>[CH1:1]=[O:2]")
for smi in ["CCO", "OCc1ccccc1"]:
    p = oxidation.RunReactants((Chem.MolFromSmiles(smi),))[0][0]
    Chem.SanitizeMol(p)
    print(smi, "->", Chem.MolToSmiles(p))
```

An amide coupling template joins any carboxylic acid with any primary or secondary amine that is not already an amide, which covers one of the most common reactions in medicinal chemistry:

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

Reaction SMILES can describe any transformation, so a gallery of named reactions becomes a few lines of data. These are written as complete, specific reactions (with `useSmiles=True`) for drawing rather than as general templates. `display()` shows each drawing inside the loop, since only the last value of a cell is shown automatically.

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

**Library enumeration** applies one template to every combination of building blocks to list all the products that could be made. Chemists use this to decide which compounds are worth making: properties can be computed for the whole virtual library, and only the most promising members are synthesised. `itertools.product` (Chapter 7) generates every pairing. A two-component reductive amination, in which an aldehyde and an amine form a new C–N bond, with 6 aldehydes and 6 amines gives 36 products:

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

Because the library is a DataFrame, it can be reshaped into an aldehyde × amine grid and shown as a heatmap, which makes it easy to see which building blocks push the lipophilicity up or down:

```{code-cell} ipython3
pivot = lib.pivot(index="aldehyde", columns="amine", values="logP")
import seaborn as sns
plt.figure(figsize=(8, 4.5))
sns.heatmap(pivot, annot=True, fmt=".1f", cmap="Spectral_r")
plt.title("cLogP across the virtual library"); plt.show()
```

## Retrosynthesis: running templates backwards

**Retrosynthesis** is the planning technique taught in organic chemistry: start from the target molecule and work backwards, "disconnecting" bonds to find simpler precursors, until you reach compounds you can buy. Each disconnection is the reverse of a known reaction. In code, a retro-template is simply a template written product-first, so applying it to the target returns candidate starting materials. Some disconnections fail to give valid molecules, so each outcome is sanitised inside `try`/`except` (Chapter 9) and invalid ones are skipped. A `set` removes duplicate suggestions. The SMARTS `-!@` means "a single bond that is not in a ring", so the Suzuki template breaks only biaryl bonds.

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

```{admonition} Summary
:class: tip
- Reaction SMILES: `reactants>agents>products`; read with `AllChem.ReactionFromSmarts(..., useSmiles=True)` and draw with `DrawReaction`.
- Atom maps (`[C:1]`) record which reactant atom becomes which product atom.
- A reaction template in SMARTS applies to any matching substrate with `RunReactants`; sanitise the products.
- Combining templates with `itertools.product` enumerates virtual libraries for property screening.
- Retro-templates run reactions backwards to suggest disconnections, the basis of computer-aided retrosynthesis.
```

## Exercises

**★ 1.** Write a reaction SMARTS for the hydrolysis of an ester and apply it to aspirin.

**★★ 2.** Enumerate all amides from four acids and four amines and plot their MW versus TPSA.

**★★★ 3.** Write a forward template for Williamson ether synthesis and a matching retro-template. Check that applying one then the other returns your starting materials.
