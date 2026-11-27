---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Organic Chemistry: Reaction Outcome, Yield and Retrosynthesis

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 22 and Part V.
```

Can a model predict what a reaction will give, how well it will work, and how to make a target in the first place? Three families of methods answer these questions: **descriptor models** on high-throughput screening data {cite:p}`ahneman2018`, **template-based** retrosynthesis with neural ranking {cite:p}`segler2018`, and **sequence-to-sequence transformers** that translate reactants into products {cite:p}`schwaller2019`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import AllChem, Draw, rdFingerprintGenerator, Crippen, Descriptors
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error

DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name)

hte = load_data("cn_coupling_hte_sim.csv")
hte.head()
```

```{admonition} About the data
:class: warning
`cn_coupling_hte_sim.csv` mimics a C–N cross-coupling screen: 8 aryl halides × 4 ligands × 5 bases × 6 additives = 960 conditions. The yields are **simulated** from a transparent logistic model (halide reactivity, ligand electron richness and cone angle, base pKa, additive inhibition) plus noise. Use it to learn the workflow; the real published screens are cited above.
```

## The design space

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
sns.boxplot(hte, x="aryl_halide", y="yield_sim", hue="aryl_halide", legend=False, ax=axes[0], palette="viridis")
axes[0].tick_params(axis="x", rotation=40); axes[0].set_title("Halide matters most")
sns.boxplot(hte, x="ligand", y="yield_sim", hue="ligand", legend=False, ax=axes[1], palette="magma")
axes[1].set_title("Ligand")
sns.boxplot(hte, x="additive", y="yield_sim", hue="additive", legend=False, ax=axes[2], palette="cividis")
axes[2].tick_params(axis="x", rotation=40); axes[2].set_title("Additives can poison the reaction")
plt.tight_layout(); plt.show()
```

```{code-cell} ipython3
pivot = hte.pivot_table(index=["aryl_halide"], columns=["ligand", "base"], values="yield_sim")
plt.figure(figsize=(11, 4))
sns.heatmap(pivot, cmap="RdYlGn", annot=False, cbar_kws={"label": "yield / %"})
plt.title("Yield landscape of the screen"); plt.tight_layout(); plt.show()
```

## Predicting yield from descriptors

Each condition becomes a feature vector: chemical descriptors of the halide plus physical parameters of ligand and base, plus one-hot additive.

```{code-cell} ipython3
def halide_features(smi):
    m = Chem.MolFromSmiles(smi)
    AllChem.ComputeGasteigerCharges(m)
    halogen = next(a for a in m.GetAtoms() if a.GetSymbol() in ("Cl", "Br", "I"))
    ipso = halogen.GetNeighbors()[0]
    return {"X_is_Cl": halogen.GetSymbol() == "Cl", "X_is_Br": halogen.GetSymbol() == "Br",
            "X_is_I": halogen.GetSymbol() == "I", "q_ipso": ipso.GetDoubleProp("_GasteigerCharge"),
            "q_X": halogen.GetDoubleProp("_GasteigerCharge"),
            "n_arom_N": sum(a.GetSymbol() == "N" and a.GetIsAromatic() for a in m.GetAtoms()),
            "logp": Crippen.MolLogP(m), "mw": Descriptors.MolWt(m)}

hf = pd.DataFrame([halide_features(s) for s in hte.aryl_halide_smiles]).astype(float)
X = pd.concat([hf, hte[["ligand_electron_richness", "ligand_cone_angle", "base_pKa"]].reset_index(drop=True),
               pd.get_dummies(hte.additive, prefix="add").astype(float)], axis=1)
y = hte.yield_sim.values
print(X.shape, "feature matrix")

rf = RandomForestRegressor(500, min_samples_leaf=2, n_jobs=-1, random_state=0)
pred_random = cross_val_predict(rf, X, y, cv=5)
pred_halide = cross_val_predict(rf, X, y, cv=GroupKFold(4), groups=hte.aryl_halide)
for name, p in [("random CV", pred_random), ("leave-halides-out CV", pred_halide)]:
    print(f"{name:22s} R² = {r2_score(y, p):.3f}  MAE = {mean_absolute_error(y, p):.1f} %")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
for ax, (name, p), c in zip(axes, [("random CV", pred_random), ("unseen halides", pred_halide)], ["#0f9d8a", "#e5487a"]):
    ax.scatter(y, p, s=12, alpha=0.6, color=c); ax.plot([0, 100], [0, 100], "k--")
    ax.set(xlabel="measured yield / %", ylabel="predicted", title=f"{name}: R² = {r2_score(y, p):.2f}", aspect="equal")
plt.tight_layout(); plt.show()
```

Predicting a **substrate the model has never seen** is far harder than interpolating within a screen. Papers that report only random splits overstate how useful a model will be on your next substrate.

```{code-cell} ipython3
rf.fit(X, y)
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
imp.tail(10).plot.barh(color="#6c4ad6", figsize=(6.5, 4)); plt.xlabel("importance")
plt.title("What the model uses"); plt.show()
```

## Condition recommendation

Given a new substrate, which conditions should you try first?

```{code-cell} ipython3
unseen = "PhBr"
train = hte.aryl_halide != unseen
rf.fit(X[train.values], y[train.values])
cand = hte[~train.values].copy()
cand["predicted"] = rf.predict(X[~train.values])
top = cand.sort_values("predicted", ascending=False).head(5)[["ligand", "base", "additive", "predicted", "yield_sim"]]
print(f"Recommendations for {unseen} (model never saw this halide):")
top.round(1)
```

```{code-cell} ipython3
best_true = cand.yield_sim.max()
got = cand.loc[top.index, "yield_sim"].max()
print(f"Best yield in the top-5 recommendations: {got:.1f} %  (best possible {best_true:.1f} %, "
      f"average condition {cand.yield_sim.mean():.1f} %)")
```

## Reaction fingerprints

A reaction can be fingerprinted as the difference between product and reactant fingerprints, which lets you cluster reaction types.

```{code-cell} ipython3
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
def reaction_fp(rxn_smiles):
    reactants, _, products = rxn_smiles.split(">")
    def fp(side):
        v = np.zeros(1024)
        for s in side.split("."):
            m = Chem.MolFromSmiles(s)
            if m: v += gen.GetCountFingerprintAsNumPy(m)
        return v
    return fp(products) - fp(reactants)

rxns = {"amidation": "OC(=O)c1ccccc1.NCc1ccccc1>>O=C(NCc1ccccc1)c1ccccc1",
        "amidation 2": "CC(=O)O.NCC>>CC(=O)NCC",
        "esterification": "CC(=O)O.OCC>>CC(=O)OCC",
        "ester 2": "OC(=O)c1ccccc1.OC>>COC(=O)c1ccccc1",
        "Suzuki": "Brc1ccccc1.OB(O)c1ccccc1>>c1ccc(-c2ccccc2)cc1",
        "Suzuki 2": "Brc1ccc(C)cc1.OB(O)c1ccncc1>>Cc1ccc(-c2ccncc2)cc1",
        "reduction": "O=Cc1ccccc1>>OCc1ccccc1",
        "reduction 2": "CC(=O)C>>CC(O)C"}
R = np.array([reaction_fp(v) for v in rxns.values()])
sim = R @ R.T / (np.linalg.norm(R, axis=1)[:, None] * np.linalg.norm(R, axis=1)[None, :])
plt.figure(figsize=(6.5, 5))
sns.heatmap(pd.DataFrame(sim, index=rxns, columns=rxns), cmap="PuOr", vmin=-1, vmax=1, square=True)
plt.title("Reaction fingerprints group reaction types"); plt.tight_layout(); plt.show()
```

Reactions of the same class sit together, which is how reaction-classification and condition-transfer models work.

## Template-based retrosynthesis with scoring

We extract templates from a small reaction set, apply them to a target, and rank the suggestions by how often the template fired in the "literature" (here, our set) and by how simple the precursors are.

```{code-cell} ipython3
TEMPLATES = {
    "amide → acid + amine": ("[C:1](=[O:2])[NX3;H1,H2:3]>>[C:1](=[O:2])O.[N:3]", 1200),
    "ester → acid + alcohol": ("[C:1](=[O:2])[OX2:3][CX4:4]>>[C:1](=[O:2])O.[O:3][C:4]", 900),
    "biaryl → halide + boronic acid": ("[c:1]-!@[c:2]>>[c:1]Br.[c:2]B(O)O", 1500),
    "aryl amine → halide + amine (C–N)": ("[c:1][NX3;H1,H2;!$(NC=O):2]>>[c:1]Br.[N:2]", 800),
    "alcohol → ketone (reduction)": ("[CX4;H1:1][OX2H:2]>>[C:1]=[O:2]", 600),
    "ether → alcohol + halide": ("[c,C:1][OX2:2][CX4:3]>>[c,C:1][O:2].[C:3]Br", 300),
}

def retro(target_smiles):
    target = Chem.MolFromSmiles(target_smiles)
    out = []
    for name, (smarts, popularity) in TEMPLATES.items():
        rxn = AllChem.ReactionFromSmarts(smarts)
        for prods in rxn.RunReactants((target,)):
            try:
                smis = []
                for p in prods:
                    Chem.SanitizeMol(p); smis.append(Chem.MolToSmiles(p))
            except Exception:
                continue
            complexity = sum(Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in smis)
            simplest = max(Chem.MolFromSmiles(s).GetNumHeavyAtoms() for s in smis)
            score = np.log(popularity) - 0.15 * simplest        # popular template, simple precursors
            out.append({"template": name, "precursors": " + ".join(sorted(set(smis))), "score": round(score, 2)})
    return pd.DataFrame(out).drop_duplicates("precursors").sort_values("score", ascending=False)

target = "CC(=O)Nc1ccc(-c2ccc(C(=O)OC)cc2)cc1"
display(Draw.MolToImage(Chem.MolFromSmiles(target), size=(420, 200), legend="target"))
routes = retro(target)
routes.head(8)
```

```{code-cell} ipython3
best = routes.iloc[0]
prec = [Chem.MolFromSmiles(s) for s in best.precursors.split(" + ")]
print("Top disconnection:", best.template)
Draw.MolsToGridImage(prec, legends=[Chem.MolToSmiles(m) for m in prec], molsPerRow=3, subImgSize=(230, 180))
```

Recursing on each precursor until everything is purchasable gives a search **tree**; real systems (Segler and co-workers {cite:p}`segler2018`) guide that search with a neural network trained on millions of reactions and a model of what is buyable.

```{code-cell} ipython3
BUYABLE = {"CC(=O)O", "Nc1ccccc1", "COC(=O)c1ccc(Br)cc1", "OB(O)c1ccc(N)cc1", "CC(=O)Nc1ccc(Br)cc1",
           "COC(=O)c1ccc(B(O)O)cc1", "Brc1ccccc1", "OC(=O)c1ccccc1", "CO"}

def plan(smiles, depth=0, max_depth=3):
    pad = "   " * depth
    if smiles in BUYABLE:
        print(f"{pad}✔ buy {smiles}")
        return True
    if depth == max_depth:
        print(f"{pad}✘ cannot route {smiles}")
        return False
    r = retro(smiles)
    if r.empty:
        print(f"{pad}✘ no template matches {smiles}")
        return False
    for _, row in r.iterrows():
        parts = row.precursors.split(" + ")
        print(f"{pad}→ {row.template}: {row.precursors}")
        if all(plan(p, depth + 1, max_depth) for p in parts):
            return True
    return False

plan(target)
```

## Exercises

**★ 1.** Which ligand and base combination gives the highest average yield across all halides?

**★★ 2.** Train the yield model with the additive column removed. How much accuracy is lost, and what does that say about the additive's role?

**★★★ 3.** Add two templates (Williamson ether synthesis, reductive amination) and plan a route to a target of your own.
