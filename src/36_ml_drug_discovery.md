---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Medicinal Chemistry: QSAR, ADMET and Virtual Screening

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Part V.
```

Drug discovery was one of the first fields to embrace machine learning. **QSAR** (quantitative structure–activity relationships) predicts potency; **ADMET** models predict absorption, distribution, metabolism, excretion and toxicity; **virtual screening** ranks millions of purchasable compounds before any are bought. Deep models have even found new antibiotic scaffolds {cite:p}`stokes2020`. The reason is economic. Bringing one drug to market takes over a decade and costs billions, most candidates fail, and every compound that is synthesised and tested costs time and money. Models that predict which compounds are likely to be potent, soluble and safe let a project spend its experiments on the most promising candidates. This chapter builds the key pieces on the book's simulated dataset: a QSAR model validated honestly with scaffold splits, a way to see which substructures drive activity, an ADMET profile, a combined score for ranking a library, a comparison of virtual-screening strategies, and a check of where the model can be trusted.

Two terms are used throughout. **Potency** is measured here as pIC₅₀ (Chapter 24): higher means less compound is needed for the effect. A **scaffold** is the core ring system of a molecule with its side chains removed (the Bemis–Murcko scaffold); medicinal chemists work in *series* of compounds that share a scaffold, so it is the natural unit for grouping molecules. RDKit computes it directly:

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
for smi in ["CC(=O)Nc1ccc(O)cc1", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "c1ccc(Cc2ccccn2)cc1"]:
    print(f"{smi:30s} scaffold {MurckoScaffold.MurckoScaffoldSmiles(smi)}")
```

Paracetamol and ibuprofen share the benzene scaffold despite looking quite different, while the third molecule's scaffold keeps both rings and the linker between them.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, rdFingerprintGenerator, Descriptors, QED
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import r2_score, roc_auc_score
from IPython.display import Image

DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name)

df = load_data("molecules_sim.csv")
mols = [Chem.MolFromSmiles(s) for s in df.smiles]
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
X = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols])
groups = df.scaffold.fillna("acyclic")
```

## A QSAR model with honest validation

A **QSAR** model is a regression model (Chapter 31) that predicts activity from structure, here a random forest on Morgan count fingerprints (Chapter 32). Its real job is to predict compounds from series the team has not yet explored, so it must be validated that way. `GroupKFold` with scaffolds as groups ensures that every molecule is predicted by a model that has never seen any molecule with the same scaffold. `cross_val_predict` returns, for every molecule, the prediction made while it was in the held-out fold, so the whole dataset gets honest out-of-sample predictions. A **hexbin** plot, which counts points in hexagonal cells, is easier to read than a scatter plot for thousands of points.

```{code-cell} ipython3
rf = RandomForestRegressor(400, min_samples_leaf=2, n_jobs=-1, random_state=0)
pred = cross_val_predict(rf, X, df.pIC50_sim, cv=GroupKFold(5), groups=groups)
fig, ax = plt.subplots(figsize=(5, 5))
hb = ax.hexbin(df.pIC50_sim, pred, gridsize=35, cmap="magma_r", mincnt=1)
ax.plot([2, 8], [2, 8], "k--"); ax.set(xlabel="true pIC50", ylabel="predicted (scaffold CV)",
                                       title=f"QSAR, R² = {r2_score(df.pIC50_sim, pred):.2f}")
plt.colorbar(hb, label="molecules"); plt.show()
```

## Which substructures drive activity?

A prediction alone does not tell a chemist what to make next; they need to know *why* the model likes a molecule. A random forest records a **feature importance** for each input, the average reduction in error from splits on that feature. The model's feature importances point to Morgan environments; mapping them back onto molecules gives medicinal chemists a picture they can act on.

```{code-cell} ipython3
rf.fit(X, df.pIC50_sim)
ao = rdFingerprintGenerator.AdditionalOutput(); ao.AllocateBitInfoMap()
top_bits = np.argsort(rf.feature_importances_)[::-1][:8]
examples = []
for b in top_bits:
    for m in mols:
        gen.GetCountFingerprint(m, additionalOutput=ao)
        if b in ao.GetBitInfoMap():
            examples.append((m, int(b), dict(ao.GetBitInfoMap())))
            break
Draw.DrawMorganBits(examples, molsPerRow=4, legends=[f"bit {b}  imp {rf.feature_importances_[b]:.3f}" for _, b, _ in examples],
                    subImgSize=(190, 160))
```

Importances describe the model as a whole. To explain one molecule, a simple and model-agnostic trick is to remove each atom in turn, by replacing it with a dummy atom, and see how much the prediction drops. Atoms whose removal lowers the predicted potency are contributing positively. The resulting per-atom weights are drawn as a colour map (Chapter 20).

```{code-cell} ipython3
def contribution_map(m):
    base = rf.predict(gen.GetCountFingerprintAsNumPy(m)[None])[0]
    w = []
    for i in range(m.GetNumAtoms()):
        em = Chem.RWMol(m); em.GetAtomWithIdx(i).SetAtomicNum(0)       # replace atom by a dummy
        w.append(base - rf.predict(gen.GetCountFingerprintAsNumPy(em.GetMol())[None])[0])
    d = rdMolDraw2D.MolDraw2DCairo(380, 300)
    SimilarityMaps.GetSimilarityMapFromWeights(m, w, colorMap="PiYG", draw2d=d)
    d.FinishDrawing()
    return Image(d.GetDrawingText()), base

best = df.pIC50_sim.idxmax()
img, p = contribution_map(mols[best]); print(df.smiles[best], f"predicted pIC50 {p:.2f}"); img
```

Green atoms raise predicted potency. Here the model has rediscovered the hinge-binding N–H next to an aromatic nitrogen that was built into the simulation.

## ADMET: multi-property profiles

Potency is not enough. A drug must be absorbed, reach its target tissue, survive metabolism long enough to act, be excreted, and not be toxic: **ADMET**. Many projects fail on these properties rather than on potency, so candidates are profiled on several at once. Some can be estimated by rules (the rule of five for oral absorption, Chapter 20; a polar surface area below about 90 Å² and a modest size for crossing the blood–brain barrier into the central nervous system), others need models, such as the toxicity classifier below, again validated by scaffold. Predicting a probability (`method="predict_proba"`) rather than a yes/no label keeps the information about how confident the model is.

```{code-cell} ipython3
def admet_flags(m):
    mw, logp = Descriptors.MolWt(m), Descriptors.MolLogP(m)
    tpsa, hbd = Descriptors.TPSA(m), Descriptors.NumHDonors(m)
    return {"oral (Ro5)": mw <= 500 and logp <= 5 and hbd <= 5,
            "CNS-penetrant (TPSA<90, MW<450)": tpsa < 90 and mw < 450,
            "soluble (logS_sim > -4)": None, "QED": QED.qed(m)}

prof = pd.DataFrame([admet_flags(m) for m in mols])
prof["soluble (logS_sim > -4)"] = df.logS_sim > -4
clf = RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=0)
p_tox = cross_val_predict(clf, X, df.toxic_sim, cv=GroupKFold(5), groups=groups, method="predict_proba")[:, 1]
print(f"toxicity model ROC-AUC (scaffold CV) = {roc_auc_score(df.toxic_sim, p_tox):.2f}")
prof["P(toxic)"] = p_tox
prof.describe().round(2)
```

## Multi-parameter optimisation and virtual screening

**Multi-parameter optimisation** (MPO) turns several property predictions into a single number for ranking. Each property is first mapped onto a **desirability** between 0 (unacceptable) and 1 (ideal) with a simple ramp: below a lower limit it scores 0, above an upper limit 1, and in between it rises linearly. A potency ramp from pIC₅₀ 5 to 7 looks like this:

```{code-cell} ipython3
for p_val in [4.5, 5.5, 6.5, 7.5]:
    print(p_val, np.clip((p_val - 5.0) / (7.0 - 5.0), 0, 1))
```

The desirabilities are combined with a **geometric mean**, the fourth root of their product, rather than an average, so that one terrible property cannot be compensated by excellent others: a desirability of 0 in any property gives a total of 0. Combine predictions into a single **desirability** score and rank the library, the way a project team would choose what to make next.

```{code-cell} ipython3
def desirability(x, lo, hi):
    return np.clip((x - lo) / (hi - lo), 0, 1)

score = (desirability(pred, 5.0, 7.0) * desirability(df.logS_sim, -5, -2) * (1 - p_tox) * prof.QED) ** (1 / 4)
df["MPO"] = score
top = df.sort_values("MPO", ascending=False).head(12)
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in top.smiles], molsPerRow=4, subImgSize=(200, 160),
                     legends=[f"MPO {r.MPO:.2f} | pIC50 {r.pIC50_sim:.1f}" for r in top.itertuples()])
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 5))
sc = ax.scatter(pred, df.logS_sim, c=p_tox, cmap="RdYlGn_r", s=10, alpha=0.7)
ax.scatter(pred[top.index], top.logS_sim, s=120, facecolor="none", edgecolor="k", lw=1.5, label="top 12 by MPO")
ax.set(xlabel="predicted pIC50", ylabel="logS (simulated)", title="Potency vs solubility, coloured by toxicity risk")
plt.colorbar(sc, label="P(toxic)"); ax.legend(); plt.show()
```

## Similarity-based screening vs. ML screening

**Virtual screening** ranks a library so that the compounds most likely to be active come first. Two strategies are compared here. *Similarity screening* (Chapter 21) needs only one known active, the query, and ranks every compound by its Tanimoto similarity to it. *Model-based screening* ranks by the QSAR prediction, which uses everything learned from all the training data. They are compared by the **enrichment factor**: the fraction of actives among the top 5 % of the ranking, divided by the fraction of actives in the whole library. An enrichment factor of 1 means the ranking is no better than random.

```{code-cell} ipython3
actives = df.index[df.pIC50_sim > 6.3]
query = actives[0]
fps = [gen.GetFingerprint(m) for m in mols]
sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[query], fps))
def enrichment(scores, top_frac=0.05):
    n = int(len(scores) * top_frac)
    hit = np.isin(np.argsort(-scores)[:n], actives).mean()
    return hit / (len(actives) / len(df))
print(f"Enrichment factor at 5 %:  similarity {enrichment(sims):.1f}x   |   QSAR model {enrichment(pred):.1f}x")
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Virtual screening does not replace experiments; it decides which experiments are worth doing. An enrichment factor of 5 means five times as many of the compounds you buy turn out to be active, which is the difference between an affordable campaign and an impossible one.
```

## Applicability domain

A model only knows the chemistry it was trained on. Flag predictions for molecules that lie far from the training set. The **applicability domain** is the region of chemical space where a model's predictions can be trusted. A common and simple measure is the Tanimoto similarity of a molecule to its nearest neighbour in the training set: a molecule with a close analogue in the training data is inside the domain, while one whose nearest neighbour has a similarity of 0.2 is effectively a new chemotype. Grouping molecules by that similarity and averaging the prediction error in each group shows whether the measure is useful: it is if the error grows as the similarity falls.

```{code-cell} ipython3
train_idx = df.index[:2000]
train_fps = [fps[i] for i in train_idx]
max_sim = np.array([max(DataStructs.BulkTanimotoSimilarity(fps[i], train_fps)) for i in df.index])
err = np.abs(pred - df.pIC50_sim)
bands = pd.cut(max_sim, [0, 0.3, 0.5, 0.7, 1.0])
pd.DataFrame({"mean |error|": err.groupby(bands, observed=True).mean(),
              "n": err.groupby(bands, observed=True).size()}).round(2)
```

```{code-cell} ipython3
plt.scatter(max_sim, err, s=6, alpha=0.4, color="#6c4ad6")
plt.axvline(0.4, ls="--", color="#e5487a")
plt.text(0.42, err.max() * 0.9, "outside the\napplicability domain", color="#e5487a", fontsize=9)
plt.xlabel("Tanimoto similarity to nearest training molecule"); plt.ylabel("|prediction error|")
plt.title("Prediction error against distance from the training data"); plt.show()
```

In this simulated dataset the trend is weak, because the simulated activity depends on a few simple motifs that even distant molecules share, so the model still does reasonably well far from its training data. On real data the effect is usually much stronger, and molecules with a nearest-neighbour similarity below about 0.4 deserve particular caution.

Report an applicability-domain check with every deployed model. A confident-looking number for a molecule unlike anything in the training set is the most dangerous output a model can give.

```{admonition} Summary
:class: tip
- QSAR models predict potency from structure; validate them with scaffold-grouped cross-validation.
- Feature importances and atom-removal maps show which substructures drive predictions.
- ADMET profiles combine rule-based filters and models; keep predicted probabilities.
- MPO maps each property onto a 0–1 desirability and combines them with a geometric mean.
- Virtual screening is judged by enrichment factors against random selection.
- Always report an applicability-domain measure such as nearest-neighbour similarity.
```

## Exercises

**★ 1.** Rank the library by predicted potency alone and compare the top 12 with the MPO top 12. What changes?

**★★ 2.** Add a synthetic-accessibility term (`rdkit.Chem.RDConfig` sascorer, or the number of rings and stereocentres) to the MPO score.

**★★★ 3.** Train the model on only the 1 000 smallest molecules and test on the largest 500. How badly does extrapolating in size hurt, and does the applicability-domain measure predict it?
