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

Matplotlib (Chapter 13) can draw anything, but statistical plots such as histograms split by category or a grid of every pair of variables take many lines of code. **Seaborn** is a library built on top of Matplotlib that draws these plots from a table in one line: you name the table, say which column goes on each axis and which column defines the colour groups, and Seaborn handles the grouping, colours, legends and statistics. **Plotly** solves a different problem. Its figures are interactive web graphics that you can zoom, rotate and hover over to read exact values, which is perfect for crowded NMR spectra, 3D energy surfaces and online books like this one. This chapter builds a small table of molecular descriptors and explores it with both libraries.

Seaborn is conventionally imported as `sns`, and `sns.set_theme` sets a consistent look for every plot that follows. RDKit (Chapter 20) is used only to compute the descriptors.

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

Seaborn works best with **tidy** data: a table in which each row is one observation (here one molecule) and each column is one variable (its class, molecular weight, logP and so on). pandas stores such a table as a `DataFrame` (Chapter 15). The simplest DataFrame is built from a dictionary of equal-length lists:

```{code-cell} ipython3
mini = pd.DataFrame({"name": ["water", "ethanol", "hexane"], "bp": [100.0, 78.4, 68.7]})
mini
```

For a more interesting dataset we compute five **descriptors** for a set of familiar compounds, grouped by class: molecular weight (MW), the octanol–water partition coefficient logP (a measure of how oily a molecule is), the topological polar surface area TPSA (the surface area of polar atoms), the number of hydrogen-bond donors and the number of rings. Each molecule becomes one dictionary, and the list of dictionaries becomes the table.

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

A **distribution** plot shows how the values of one variable are spread: where they cluster, how wide the spread is, and whether there are outliers. A histogram counts values in bins; a violin plot draws a smoothed outline of the distribution for each category so that categories can be compared side by side. In Seaborn you pass the table, the column for `x` (and `y`), and optionally `hue=`, a column whose categories get different colours. The simplest case is one histogram:

```{code-cell} ipython3
sns.histplot(df, x="MW", bins=10)
plt.show()
```

Adding `hue="class"` and stacking the classes shows that hydrocarbons have high logP and acids and alcohols low logP, while the violin plot shows that the drugs and acids have the largest polar surface areas:

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
sns.histplot(df, x="logP", hue="class", multiple="stack", ax=axes[0])
sns.violinplot(df, x="class", y="TPSA", hue="class", inner="point", ax=axes[1])
axes[1].tick_params(axis="x", rotation=30)
plt.tight_layout(); plt.show()
```

## Relationships

A **relational** plot shows how two variables change together. `sns.relplot` is a scatter plot that can map further columns to colour (`hue`) and marker size (`size`), so a single figure shows four variables at once. Here the size of each point is its polar surface area, and the classes separate clearly: at the same molecular weight, hydrocarbons are far more lipophilic than acids.

```{code-cell} ipython3
sns.relplot(df, x="MW", y="logP", hue="class", size="TPSA", sizes=(30, 300), height=4.5, aspect=1.4)
plt.title("Polarity versus size")
plt.show()
```

A **pair plot** goes further and draws a scatter plot for every pair of numeric columns, with each column's distribution on the diagonal. It is the standard first look at a new dataset because it reveals every pairwise relationship at a glance; `corner=True` omits the duplicate upper half.

```{code-cell} ipython3
sns.pairplot(df[["MW", "logP", "TPSA", "HBD", "class"]], hue="class", corner=True, height=1.9)
plt.show()
```

## Heatmaps

A **heatmap** shows a table of numbers as a grid of coloured cells. Its most common use in data analysis is to display a **correlation matrix**. The correlation coefficient $r$ between two variables runs from −1 (one goes down exactly as the other goes up) through 0 (no linear relationship) to +1 (they rise together). `df.corr()` computes $r$ for every pair of columns, and a diverging colormap centred on zero makes positive and negative correlations easy to tell apart. `annot=True` writes the number in each cell.

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

`sns.regplot` draws a scatter plot together with a fitted regression line and a shaded confidence band, which shows how certain the fit is. `order=2` fits a quadratic instead of a straight line. This is a quick visual check of a trend; for the actual fitted parameters and their uncertainties, use SciPy (Chapter 16) or scikit-learn (Chapter 31). The boiling points of the n-alkanes rise with chain length but flatten out, so a curve fits better than a line.

```{code-cell} ipython3
alkanes = pd.DataFrame({"n_C": range(1, 11),
                        "bp_C": [-161.5, -88.6, -42.1, -0.5, 36.1, 68.7, 98.4, 125.7, 150.8, 174.1]})
sns.regplot(alkanes, x="n_C", y="bp_C", order=2, color="#6c4ad6")
plt.xlabel("Number of carbons"); plt.ylabel("Boiling point / °C"); plt.title("n-Alkane boiling points")
plt.show()
```

## Interactive plots with Plotly

Plotly figures are drawn by your web browser rather than as fixed images, so you can zoom by dragging a box, pan, hide a series by clicking its legend entry, and hover over any point to see its exact values. The `plotly.express` module (imported as `px`) offers one-line functions that take a DataFrame and column names, just like Seaborn. A first interactive line needs only lists:

```{code-cell} ipython3
import plotly.express as px
px.line(x=[0, 5, 10, 15, 20, 25], y=[2.9, 4.1, 4.8, 5.4, 8.7, 11.3],
        labels={"x": "V(NaOH) / mL", "y": "pH"}, markers=True, title="Hover over the points").show()
```

With the descriptor table, `hover_data=["smiles"]` makes each point show its molecule's SMILES, turning the plot into a small explorer of chemical space.

```{code-cell} ipython3
import plotly.express as px
fig = px.scatter(df, x="MW", y="logP", color="class", size="TPSA", hover_data=["smiles"],
                 title="Interactive chemical space (hover me)", template="plotly_white")
fig.show()
```

### An interactive spectrum

For full control Plotly has a lower-level interface, `plotly.graph_objects` (imported as `go`), in which you build a `Figure` from *traces* such as `go.Scatter`. An NMR spectrum is a good use: the multiplets are too narrow to see at full width but resolve beautifully when you zoom in. Each peak is modelled as a **Lorentzian** line shape, and spin–spin coupling splits the CH₃ into a 1:2:1 triplet and the CH₂ into a 1:3:3:1 quartet, following the n + 1 rule. The x axis is reversed because chemical shifts are conventionally plotted from high to low ppm.

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

A **potential energy surface** gives the energy of a system as a function of its geometry; minima are stable structures and the low passes between them are transition states. Real molecules have many coordinates, but a two-coordinate model can be drawn as a landscape. The Müller–Brown surface is a classic two-dimensional model of a reaction landscape with three minima, built from a sum of four Gaussian-like terms. `go.Surface` draws it in 3D, and you can drag to rotate it. Energies above 100 are clipped so the minima stay visible.

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

Plotly can also animate a figure over a variable such as time. You supply one table with a column that labels each frame (`animation_frame="t"`), and Plotly adds a play button and a slider. Here the concentrations of the consecutive reaction A → B → C are computed at 40 times from the analytical solution (Chapter 25), and the bars show the intermediate B rising and then falling.

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

```{admonition} Summary
:class: tip
- Seaborn plots tidy tables in one line: pass the DataFrame, column names for `x`/`y`, and `hue` for colour groups.
- Distributions: `histplot`, `violinplot`, `boxplot`. Relationships: `relplot`, `pairplot`, `regplot`. Matrices: `heatmap`.
- A correlation heatmap reveals duplicated information between descriptors.
- Plotly figures are interactive: `plotly.express` for quick plots, `graph_objects` for full control, `animation_frame` for animations.
```

## Exercises

**★ 1.** Use `sns.boxplot` to compare molecular weights across classes.

**★★ 2.** Make an interactive Plotly line chart of the IR spectrum from Chapter 12.

**★★★ 3.** Plot the Müller–Brown surface as a Plotly contour map and mark its three minima, found with `scipy.optimize.minimize` (Chapter 16).
