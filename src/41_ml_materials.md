---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Materials Science: Property Prediction and Discovery

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 30 and Part V.
```

Materials informatics predicts the properties of solids from composition and structure, then uses those models to screen millions of hypothetical compounds. Composition-only models {cite:p}`ward2016`, crystal graph networks {cite:p}`xie2018` and open databases {cite:p}`jain2013` are the standard toolkit; large-scale screening of this kind has recently proposed hundreds of thousands of new stable crystals {cite:p}`merchant2023`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from pymatgen.core import Composition, Element

DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name, **kw):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name, **kw)

# keep_default_na=False matters here: the formula "NaN" (sodium nitride, Na + N) would
# otherwise be read as a missing value. Chemistry breaks defaults.
mat = load_data("binary_materials_sim.csv", keep_default_na=False, na_values=[""])
mat.head()
```

```{admonition} About the data
:class: warning
`binary_materials_sim.csv` lists 770 real binary compositions, but the band gaps and formation energies are **simulated** from electronegativity differences, radii and charge balance, plus noise. Use it to learn the workflow; for real values use the Materials Project {cite:p}`jain2013` or OQMD.
```

## Exploring the dataset

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sns.scatterplot(mat, x="chi_B", y="chi_A", hue="band_gap_sim_eV", palette="viridis", size="band_gap_sim_eV",
                sizes=(5, 70), ax=axes[0])
axes[0].set(xlabel="χ(anion)", ylabel="χ(cation)", title="Band gap tracks electronegativity difference")
sns.histplot(mat, x="band_gap_sim_eV", hue="charge_balanced", multiple="stack", bins=30, ax=axes[1])
axes[1].set_title("Charge-balanced compounds are the insulators")
sns.scatterplot(mat, x="formation_energy_sim_eV_atom", y="band_gap_sim_eV", hue="charge_balanced", s=18, ax=axes[2])
axes[2].set(xlabel="formation energy / eV atom$^{-1}$", title="Stability vs gap")
plt.tight_layout(); plt.show()
```

## Composition featurisation

```{code-cell} ipython3
PROPS = {"Z": lambda e: e.Z, "X": lambda e: e.X, "row": lambda e: e.row, "group": lambda e: e.group,
         "radius": lambda e: float(e.atomic_radius or 0), "mass": lambda e: float(e.atomic_mass),
         "mendeleev": lambda e: e.mendeleev_no, "valence": lambda e: sum(e.full_electronic_structure[-1][2:])}

def featurise(formula):
    c = Composition(formula)
    fr = np.array([c.get_atomic_fraction(e) for e in c.elements])
    out = {}
    for name, f in PROPS.items():
        v = np.array([f(e) for e in c.elements], dtype=float)
        mean = float((fr * v).sum())
        out[f"mean_{name}"] = mean
        out[f"range_{name}"] = float(v.max() - v.min())
        out[f"dev_{name}"] = float((fr * np.abs(v - mean)).sum())
        out[f"max_{name}"], out[f"min_{name}"] = float(v.max()), float(v.min())
    out["n_elements"] = len(c.elements)
    out["mean_stoich"] = float(np.mean(list(c.as_dict().values())))
    return out

F = pd.DataFrame([featurise(f) for f in mat.formula])
print(F.shape, "features from composition alone")
F.head(3).round(2).iloc[:, :8]
```

## Predicting band gap and formation energy

```{code-cell} ipython3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_predict, KFold, GroupKFold
from sklearn.metrics import mean_absolute_error, r2_score

cv = KFold(5, shuffle=True, random_state=0)
targets = {"band_gap_sim_eV": "band gap / eV", "formation_energy_sim_eV_atom": "formation energy / eV atom$^{-1}$"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
models = {}
for ax, (col, label) in zip(axes, targets.items()):
    rf = RandomForestRegressor(400, n_jobs=-1, random_state=0)
    p = cross_val_predict(rf, F.values, mat[col], cv=cv)
    models[col] = rf.fit(F.values, mat[col])
    ax.scatter(mat[col], p, s=12, alpha=0.6, color="#6c4ad6")
    lo, hi = mat[col].min(), mat[col].max()
    ax.plot([lo, hi], [lo, hi], "k--")
    ax.set(xlabel=f"true {label}", ylabel="predicted", aspect="equal",
           title=f"MAE = {mean_absolute_error(mat[col], p):.3f}, R² = {r2_score(mat[col], p):.2f}")
plt.tight_layout(); plt.show()
```

### Extrapolating to unseen chemistries

```{code-cell} ipython3
for col in targets:
    p_rand = cross_val_predict(RandomForestRegressor(300, n_jobs=-1, random_state=0), F.values, mat[col], cv=cv)
    p_group = cross_val_predict(RandomForestRegressor(300, n_jobs=-1, random_state=0), F.values, mat[col],
                                cv=GroupKFold(5), groups=mat.A)
    print(f"{col:32s} random CV MAE {mean_absolute_error(mat[col], p_rand):.3f}   "
          f"leave-cation-out MAE {mean_absolute_error(mat[col], p_group):.3f}")
```

Leaving out whole element families is much harder, and it is the test that matters if you want to predict genuinely new materials.

```{code-cell} ipython3
imp = pd.Series(models["band_gap_sim_eV"].feature_importances_, index=F.columns).sort_values()
imp.tail(12).plot.barh(color="#0f9d8a", figsize=(7, 4)); plt.xlabel("importance")
plt.title("The model relies on electronegativity spread and radii"); plt.show()
```

## A crystal graph network, in miniature

Crystal graph convolutional networks {cite:p}`xie2018` build a graph whose nodes are atoms in the unit cell and whose edges are neighbours within a cutoff, with distances as edge features. Here is the featurisation step for a real structure.

```{code-cell} ipython3
from pymatgen.core import Structure, Lattice

srtio3 = Structure(Lattice.cubic(3.905), ["Sr", "Ti", "O", "O", "O"],
                   [[0, 0, 0], [.5, .5, .5], [.5, .5, 0], [.5, 0, .5], [0, .5, .5]])

def crystal_graph(structure, cutoff=4.0, n_gauss=10):
    nodes = np.array([[s.specie.Z, s.specie.X, float(s.specie.atomic_radius or 0), s.specie.row, s.specie.group]
                      for s in structure])
    centres = np.linspace(0.5, cutoff, n_gauss)
    edges, edge_feats = [], []
    for i, site in enumerate(structure):
        for nb in structure.get_neighbors(site, cutoff):
            edges.append((i, nb.index))
            edge_feats.append(np.exp(-((nb.nn_distance - centres) / 0.5) ** 2))     # Gaussian distance expansion
    return nodes, np.array(edges), np.array(edge_feats)

nodes, edges, efeat = crystal_graph(srtio3)
print(f"{len(nodes)} nodes, {len(edges)} directed edges, edge features {efeat.shape}")
fig, axes = plt.subplots(1, 2, figsize=(11, 3.2), gridspec_kw={"width_ratios": [1, 2]})
sns.heatmap(nodes, ax=axes[0], cmap="Purples", cbar=False, yticklabels=[str(s.specie) for s in srtio3],
            xticklabels=["Z", "χ", "r", "row", "group"])
axes[0].set_title("node features")
sns.heatmap(efeat[:20], ax=axes[1], cmap="Greens", cbar=False)
axes[1].set(title="Gaussian-expanded bond distances (first 20 edges)", xlabel="distance basis function")
plt.tight_layout(); plt.show()
```

The convolution then updates each atom from its neighbours and their bond distances, exactly like the molecular message-passing network of Chapter 34, and pools over the unit cell to predict a per-atom property.

## Screening for stability: the convex hull

A compound is thermodynamically stable only if it lies **on** the convex hull of formation energies, not above it.

```{code-cell} ipython3
from scipy.spatial import ConvexHull
rng = np.random.default_rng(0)
x = np.array([0, 0.25, 0.33, 0.4, 0.5, 0.5, 0.6, 0.67, 0.75, 1.0])         # composition A_x B_(1-x)
Ef = np.array([0, -0.35, -0.52, -0.46, -0.78, -0.60, -0.42, -0.55, -0.30, 0])
pts = np.column_stack([x, Ef])
hull = ConvexHull(np.vstack([pts, [[0.5, 1.0]]]))                            # add a high point to close the hull
lower = sorted({v for s in hull.simplices for v in s if v < len(pts) and Ef[v] <= 0}, key=lambda i: x[i])
lower = [i for i in lower if Ef[i] == min(Ef[j] for j in lower if np.isclose(x[j], x[i]))]

plt.figure(figsize=(7, 4.5))
plt.plot(x[lower], Ef[lower], "-o", color="#0f9d8a", lw=2.5, ms=10, label="on the hull (stable)")
above = [i for i in range(len(x)) if i not in lower]
plt.scatter(x[above], Ef[above], color="#e5487a", s=70, zorder=5, label="above the hull (metastable)")
for i in above:
    e_hull = np.interp(x[i], x[lower], Ef[lower])
    plt.vlines(x[i], e_hull, Ef[i], color="#e5487a", ls=":")
    plt.text(x[i] + 0.012, (Ef[i] + e_hull) / 2, f"+{Ef[i] - e_hull:.2f} eV", fontsize=8, color="#e5487a")
plt.axhline(0, color="k", lw=0.8)
plt.xlabel("x in A$_x$B$_{1-x}$"); plt.ylabel("formation energy / eV atom$^{-1}$")
plt.legend(); plt.title("Energy above the hull decides synthesisability"); plt.show()
```

## A screening campaign

Predict properties for every composition the model has **not** seen, then filter for a target window: a stable semiconductor with a gap between 1.1 and 1.7 eV, the range for a single-junction solar absorber.

```{code-cell} ipython3
import itertools
cations = ["Li", "Na", "K", "Mg", "Ca", "Sr", "Zn", "Cd", "Al", "Ga", "In", "Be", "Ba", "Rb"]
anions = ["O", "S", "Se", "Te", "N", "P", "As", "F", "Cl", "Br", "I", "Sb"]
cand = []
for a, b in itertools.product(cations, anions):
    for na, nb in [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)]:
        f = f"{a}{na if na > 1 else ''}{b}{nb if nb > 1 else ''}"
        if f not in set(mat.formula):
            cand.append(f)
cand = pd.DataFrame({"formula": cand})
Fc = pd.DataFrame([featurise(f) for f in cand.formula])[F.columns]
cand["gap"] = models["band_gap_sim_eV"].predict(Fc.values)
cand["Ef"] = models["formation_energy_sim_eV_atom"].predict(Fc.values)
hits = cand.query("1.1 <= gap <= 1.7 and Ef < -0.4").sort_values("Ef")
print(f"{len(cand)} candidates screened, {len(hits)} in the target window")

fig, ax = plt.subplots(figsize=(7.5, 5))
ax.scatter(cand.Ef, cand.gap, s=8, color="lightgrey")
ax.add_patch(plt.Rectangle((-3, 1.1), 2.6, 0.6, color="#0f9d8a", alpha=0.15))
ax.scatter(hits.Ef, hits.gap, s=60, color="#0f9d8a", edgecolor="k")
for r in hits.head(8).itertuples():
    ax.annotate(r.formula, (r.Ef, r.gap), textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set(xlabel="predicted formation energy / eV atom$^{-1}$", ylabel="predicted band gap / eV",
       title="Screening for a solar absorber (teal window)")
plt.show()
hits.head(10).round(2)
```

```{warning}
Composition-only models cannot distinguish polymorphs: anatase and rutile TiO₂ have the same formula and very different properties. They are a filter, not an oracle. The next step is always a structure prediction and a DFT calculation on the survivors.
```

## Exercises

**★ 1.** Which three features are most important for formation energy? Do they make physical sense?

**★★ 2.** Retrain the band-gap model without any electronegativity features. How much accuracy is lost?

**★★★ 3.** Add ternary compositions (A–B–O) to the screen and plot the results on a ternary diagram with `pymatgen.util.plotting`.
