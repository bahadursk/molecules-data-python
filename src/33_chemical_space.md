---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Mapping Chemical Space

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 21, 31 and 32.
```

"Chemical space" is the set of all possible molecules, estimated at 10⁶⁰ drug-sized compounds. We cannot list it, but we can **map** the part we have: compress fingerprints into two dimensions, cluster similar molecules, and spot gaps worth exploring. A fingerprint places each molecule as a point in a space with thousands of dimensions, one per bit, and molecules that share substructures lie close together in it. Nobody can look at 2048 dimensions, so **dimensionality reduction** squeezes that space into two dimensions that can be plotted while keeping as much of the neighbourhood structure as possible. **Clustering** groups similar molecules automatically. These tools are how chemists answer practical questions: How diverse is my library? Which regions are crowded and which are empty? Does activity cluster in one part of space? Is my test set genuinely different from my training set?

We work with 1200 molecules from the book's dataset and their Morgan fingerprints (Chapter 21), stored both as RDKit bit vectors (for Tanimoto calculations) and as a NumPy matrix (for scikit-learn).

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator, Draw
from rdkit.ML.Cluster import Butina
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans

DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name)

df = load_data("molecules_sim.csv").sample(1200, random_state=0).reset_index(drop=True)
mols = [Chem.MolFromSmiles(s) for s in df.smiles]
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
fps = [gen.GetFingerprint(m) for m in mols]
X = np.array([gen.GetFingerprintAsNumPy(m) for m in mols], dtype=float)
```

## PCA: the linear view

**Principal component analysis** (PCA) finds new axes, called principal components, that are combinations of the original features. The first component points in the direction along which the data vary most, the second in the direction of greatest remaining variation at right angles to the first, and so on. Projecting the data onto the first two components gives the most "spread out" flat picture possible with straight-line axes. The **explained variance ratio** says what fraction of the total variation each component captures, so you can see how much is lost by keeping only two.

PCA is easiest to understand on data whose shape we know. Points scattered along a diagonal line in two dimensions have almost all their variation along one direction, and PCA finds it:

```{code-cell} ipython3
rng_demo = np.random.default_rng(0)
t_demo = rng_demo.normal(0, 1, 200)
pts = np.column_stack([t_demo, 0.5 * t_demo + rng_demo.normal(0, 0.1, 200)])
p2 = PCA(2).fit(pts)
print("variance explained:", p2.explained_variance_ratio_.round(3))
print("first component direction:", p2.components_[0].round(2))
```

For fingerprints the picture is very different: the variance is spread across many components, so two dimensions capture only a small part of it, though the map still shows a smooth trend in logP.

```{code-cell} ipython3
pca = PCA(n_components=10).fit(X)
P = pca.transform(X)
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
axes[0].bar(range(1, 11), pca.explained_variance_ratio_ * 100, color="#6c4ad6")
axes[0].set(xlabel="component", ylabel="% variance", title="Fingerprint space is high-dimensional")
sc = axes[1].scatter(P[:, 0], P[:, 1], c=df.logP, cmap="viridis", s=8)
plt.colorbar(sc, ax=axes[1], label="logP"); axes[1].set(xlabel="PC1", ylabel="PC2", title="PCA map coloured by logP")
plt.tight_layout(); plt.show()
```

## t-SNE: the neighbourhood view

**t-SNE** (t-distributed stochastic neighbour embedding) takes a different approach. Instead of preserving overall variance with straight axes, it tries to place molecules that are neighbours in the high-dimensional space as neighbours on the map, bending and stretching the space as needed. The result shows clusters of related molecules much more clearly than PCA. The price is that only local structure is meaningful: the distance between two far-apart clusters, and the size of a cluster, carry little information. The `perplexity` parameter roughly sets how many neighbours each point "pays attention to". Using the **Jaccard** metric, which is one minus the Tanimoto similarity, makes the map respect the same similarity chemists use. Colouring by **scaffold** (the ring system at the core of each molecule, Chapter 36) shows that molecules built on the same core form tight groups.

```{code-cell} ipython3
emb = TSNE(n_components=2, metric="jaccard", init="random", perplexity=35, random_state=0).fit_transform(X.astype(bool))
top = df.scaffold.value_counts().index[:8]
colours = dict(zip(top, plt.cm.tab10.colors))
fig, ax = plt.subplots(figsize=(8, 7))
other = ~df.scaffold.isin(top)
ax.scatter(emb[other, 0], emb[other, 1], s=6, color="lightgrey", label="other scaffolds")
for s in top:
    m = df.scaffold == s
    ax.scatter(emb[m, 0], emb[m, 1], s=12, color=colours[s], label=s[:28])
ax.legend(fontsize=7, markerscale=2, loc="best"); ax.set_title("t-SNE of Morgan fingerprints (Jaccard), by scaffold"); ax.axis("off")
plt.show()
```

```{code-cell} ipython3
Draw.MolsToGridImage([Chem.MolFromSmiles(s) if s else Chem.MolFromSmiles("C") for s in top], legends=[f"scaffold {i+1}" for i in range(8)],
                     molsPerRow=4, subImgSize=(180, 140))
```

UMAP (`pip install umap-learn`) gives similar maps faster and preserves more global structure; the call is `umap.UMAP(metric="jaccard").fit_transform(X)`.

## Property landscapes

Colouring the same map by a property turns it into a **landscape**. If the property changes smoothly across the map, similar molecules have similar values, and a model based on similarity will predict it well. If neighbouring points have very different colours, the property is rugged and hard to learn.

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, col, cmap in zip(axes, ["pIC50_sim", "logS_sim", "toxic_sim"], ["plasma", "viridis", "coolwarm"]):
    sc = ax.scatter(emb[:, 0], emb[:, 1], c=df[col], s=7, cmap=cmap)
    plt.colorbar(sc, ax=ax); ax.set_title(col); ax.axis("off")
plt.tight_layout(); plt.show()
```

Activity forms "islands" in chemical space. The steeper the landscape (similar molecules with very different activities, called **activity cliffs**), the harder it is to model.

## Butina clustering

**Clustering** divides a dataset into groups of similar items without being told what the groups are (this is *unsupervised* learning). For molecules the natural measure is Tanimoto distance, $1 - T$. Butina's algorithm {cite:p}`butina1999` groups molecules whose Tanimoto distance to a cluster centre is below a threshold. It works by counting, for each molecule, how many others lie within the threshold; the molecule with the most neighbours becomes the first cluster centre and takes those neighbours into its cluster, and the process repeats with the remaining molecules. The threshold controls how tight clusters are: 0.6 corresponds to a similarity of 0.4, a common choice for Morgan fingerprints. The function needs the distances as a flat list of the lower triangle of the distance matrix, which the loop builds.

```{code-cell} ipython3
dists = []
for i in range(1, len(fps)):
    sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
    dists.extend(1 - s for s in sims)
clusters = Butina.ClusterData(dists, len(fps), 0.6, isDistData=True)
sizes = sorted((len(c) for c in clusters), reverse=True)
print(len(clusters), "clusters; largest:", sizes[:10])
centroids = [mols[c[0]] for c in sorted(clusters, key=len, reverse=True)[:8]]
Draw.MolsToGridImage(centroids, legends=[f"cluster {i+1} (n={s})" for i, s in enumerate(sizes[:8])], molsPerRow=4, subImgSize=(190, 150))
```

## Diversity picking

When only a few compounds can be tested, it is better to choose ones that cover the space than ones that are all alike. **MaxMin picking** chooses a diverse subset greedily: start with one molecule, then repeatedly add the molecule whose distance to its nearest already-picked neighbour is largest, so each new pick is as different as possible from everything chosen so far. To choose 12 molecules that cover the space for screening, use MaxMin picking:

```{code-cell} ipython3
from rdkit.SimDivFilters.rdSimDivPickers import MaxMinPicker
picks = list(MaxMinPicker().LazyBitVectorPick(fps, len(fps), 12, seed=42))
fig, ax = plt.subplots(figsize=(7, 6))
ax.scatter(emb[:, 0], emb[:, 1], s=5, color="lightgrey")
ax.scatter(emb[picks, 0], emb[picks, 1], s=120, color="#e5487a", edgecolor="k", marker="*")
ax.set_title("12 diverse picks spread across the map"); ax.axis("off"); plt.show()
```

## Scaffold splits: an honest test

Chapter 31 split data at random. For molecules this can be misleading. A random split puts close analogues in both training and test sets, which flatters the model. A **scaffold split** keeps whole scaffolds out of training, mimicking the prediction of genuinely new chemistry. scikit-learn's `GroupShuffleSplit` does this when each molecule's group is its scaffold: all molecules of a group go to the same side of the split. Molecules with no rings have no scaffold, so they are given a shared label first. Repeating each split with five seeds shows how stable the result is.

```{code-cell} ipython3
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupShuffleSplit, ShuffleSplit

y = df.pIC50_sim.values
groups = df.scaffold.fillna("acyclic").values
scores = {"random split": [], "scaffold split": []}
for seed in range(5):
    for name, splitter in [("random split", ShuffleSplit(1, test_size=0.2, random_state=seed)),
                           ("scaffold split", GroupShuffleSplit(1, test_size=0.2, random_state=seed))]:
        tr, te = next(splitter.split(X, y, groups))
        m = RandomForestRegressor(200, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
        scores[name].append(r2_score(y[te], m.predict(X[te])))
for k, v in scores.items():
    print(f"{k:15s} R² = {np.mean(v):.2f} ± {np.std(v):.2f}")
```

The scaffold split gives a much lower R², and that lower number is the honest estimate of how the model would perform on a new chemical series.

```{admonition} 🤖 ML corner
:class: ml-corner
Always report how you split your data. Benchmark suites such as MoleculeNet {cite:p}`wu2018` recommend scaffold or time splits for exactly this reason.
```

```{admonition} Summary
:class: tip
- Fingerprints place molecules in a high-dimensional space; dimensionality reduction draws it in 2D.
- PCA uses straight axes of maximum variance; check the explained variance.
- t-SNE (and UMAP) preserve neighbourhoods: trust clusters, not distances between them.
- Colouring maps by properties reveals smooth regions and activity cliffs.
- Butina clustering groups molecules by Tanimoto distance; MaxMin picking chooses diverse subsets.
- Scaffold splits test generalisation to new chemistry more honestly than random splits.
```

## Exercises

**★ 1.** Colour the PCA map by molecular weight. Which component tracks size?

**★★ 2.** Use k-means with 10 clusters on the PCA coordinates and draw one molecule from each cluster.

**★★★ 3.** Find activity cliffs: pairs with Tanimoto > 0.7 but |ΔpIC50| > 1.5. Draw the three most extreme pairs side by side.
