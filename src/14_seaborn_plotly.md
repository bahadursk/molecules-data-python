---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Seaborn and Plotly: Statistical and Interactive Graphics

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 12–13. pandas (Chapter 15) is used lightly.
```

**Seaborn** draws attractive statistical plots from tables in one line. **Plotly** makes figures you can zoom, rotate and hover over, perfect for spectra and energy surfaces in an online book.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors

sns.set_theme(style="whitegrid", palette="Set2")
```

## A small molecule dataset

We compute descriptors for a set of familiar compounds, grouped by class.

```{code-cell} ipython3
data = {
    "alcohol": ["CO", "CCO", "CCCO", "CCCCO", "CC(C)O", "OCCO", "OCC(O)CO", "c1ccccc1CO"],
    "acid": ["C(=O)O", "CC(=O)O", "CCC(=O)O", "OC(=O)c1ccccc1", "OC(=O)CC(O)(CC(=O)O)C(=O)O", "CC(O)C(=O)O"],
    "amine": ["CN", "CCN", "CCCCN", "c1ccccc1N", "NCCN", "CN(C)C", "C1CCNCC1"],
    "hydrocarbon": ["CCCCCC", "c1ccccc1", "Cc1ccccc1", "C1CCCCC1", "CC=CC", "c1ccc2ccccc2c1", "CCCCCCCC"],
    "drug": ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CC(=O)Nc1ccc(O)cc1",
             "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "CN1CCC[C@H]1c1cccnc1"],
}
rows = []
for cls, smiles in data.items():
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        rows.append({"class": cls, "smiles": s, "MW": Descriptors.MolWt(m), "logP": Crippen.MolLogP(m),
                     "TPSA": rdMolDescriptors.CalcTPSA(m), "HBD": rdMolDescriptors.CalcNumHBD(m),
                     "rings": rdMolDescriptors.CalcNumRings(m)})
df = pd.DataFrame(rows)
df.head()
```

## Distributions

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
sns.histplot(df, x="logP", hue="class", multiple="stack", ax=axes[0])
sns.violinplot(df, x="class", y="TPSA", hue="class", inner="point", ax=axes[1])
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()
```

## Relationships

```{code-cell} ipython3
sns.relplot(df, x="MW", y="logP", hue="class", size="TPSA", sizes=(30, 300), height=4.5, aspect=1.4)
plt.title("Polarity versus size")
plt.show()
```

```{code-cell} ipython3
sns.pairplot(df[["MW", "logP", "TPSA", "HBD", "class"]], hue="class", corner=True, height=1.9)
plt.show()
```

## Heatmaps

```{code-cell} ipython3
corr = df[["MW", "logP", "TPSA", "HBD", "rings"]].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", vmin=-1, vmax=1, square=True)
plt.title("Descriptor correlations")
plt.show()
```

```{admonition} 🤖 ML corner
:class: ml-corner
Strongly correlated descriptors carry duplicate information. Spotting them in a heatmap is the first step of feature selection (Chapter 32).
```

## Regression plots

```{code-cell} ipython3
alkanes = pd.DataFrame({"n_C": range(1, 11),
                        "bp_C": [-161.5, -88.6, -42.1, -0.5, 36.1, 68.7, 98.4, 125.7, 150.8, 174.1]})
sns.regplot(alkanes, x="n_C", y="bp_C", order=2, color="#6c4ad6")
plt.xlabel("Number of carbons"); plt.ylabel("Boiling point / °C"); plt.title("n-Alkane boiling points")
plt.show()
```

## Interactive plots with Plotly

Hover over the points to see each molecule's SMILES.

```{code-cell} ipython3
import plotly.express as px
fig = px.scatter(df, x="MW", y="logP", color="class", size="TPSA", hover_data=["smiles"],
                 title="Interactive chemical space (hover me)", template="plotly_white")
fig.show()
```

### An interactive spectrum

```{code-cell} ipython3
import plotly.graph_objects as go
shift = np.linspace(0, 10, 4000)

def lorentz(x, x0, w=0.01):
    return w**2 / ((x - x0) ** 2 + w**2)

# Ethyl acetate 1H NMR: CH3 triplet 1.26, CH3 singlet 2.05, CH2 quartet 4.12 ppm (J = 7.1 Hz at 400 MHz)
J = 7.1 / 400
spec = 3 * lorentz(shift, 2.05)
spec += sum(c * lorentz(shift, 1.26 + k * J) for c, k in zip([1, 2, 1], [-1, 0, 1]))
spec += sum(c * 0.66 * lorentz(shift, 4.12 + k * J) for c, k in zip([1, 3, 3, 1], [-1.5, -0.5, 0.5, 1.5]))
fig = go.Figure(go.Scatter(x=shift, y=spec, line=dict(color="#0f9d8a")))
fig.update_layout(title="Simulated ¹H NMR of ethyl acetate (zoom into the multiplets)",
                  xaxis=dict(title="δ / ppm", autorange="reversed"), yaxis_title="Intensity", template="plotly_white")
fig.show()
```

### A 3D potential energy surface

The Müller–Brown surface is a classic two-dimensional model of a reaction landscape with three minima.

```{code-cell} ipython3
A = [-200, -100, -170, 15]; a = [-1, -1, -6.5, 0.7]; b = [0, 0, 11, 0.6]; c = [-10, -10, -6.5, 0.7]
x0 = [1, 0, -0.5, -1]; y0 = [0, 0.5, 1.5, 1]
X, Y = np.meshgrid(np.linspace(-1.5, 1.2, 120), np.linspace(-0.3, 2.0, 120))
V = sum(A[k] * np.exp(a[k] * (X - x0[k])**2 + b[k] * (X - x0[k]) * (Y - y0[k]) + c[k] * (Y - y0[k])**2) for k in range(4))
V = np.clip(V, None, 100)
fig = go.Figure(go.Surface(x=X, y=Y, z=V, colorscale="Viridis"))
fig.update_layout(title="Müller–Brown potential energy surface (drag to rotate)", height=500,
                  scene=dict(zaxis_title="E"))
fig.show()
```

### An animated reaction

```{code-cell} ipython3
t = np.linspace(0, 10, 40)
k1, k2 = 0.8, 0.3
frames = []
for ti in t:
    A_ = np.exp(-k1 * ti)
    B_ = k1 / (k2 - k1) * (np.exp(-k1 * ti) - np.exp(-k2 * ti))
    frames.append(pd.DataFrame({"species": ["A", "B", "C"], "conc": [A_, B_, 1 - A_ - B_], "t": round(ti, 2)}))
anim = pd.concat(frames)
px.bar(anim, x="species", y="conc", color="species", animation_frame="t", range_y=[0, 1],
       title="A → B → C (press play)", color_discrete_sequence=["#e5487a", "#f2a93b", "#0f9d8a"]).show()
```

## Exercises

**★ 1.** Use `sns.boxplot` to compare molecular weights across classes.

**★★ 2.** Make an interactive Plotly line chart of the IR spectrum from Chapter 12.

**★★★ 3.** Plot the Müller–Brown surface as a Plotly contour map and mark its three minima, found with `scipy.optimize.minimize` (Chapter 16).
