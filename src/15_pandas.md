---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# pandas: Tables of Molecules and Measurements

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 1–12.
```

Most chemical data is a table: compounds in rows, properties in columns. **pandas** is Python's spreadsheet, with labelled rows and columns, fast filtering, grouping and joining.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Series and DataFrames

```{code-cell} ipython3
bp = pd.Series({"water": 100.0, "ethanol": 78.4, "acetone": 56.1, "hexane": 68.7}, name="bp_C")
print(bp, "\n")
print(bp["ethanol"], bp.max(), bp.idxmin())
```

```{code-cell} ipython3
elements = pd.DataFrame({
    "symbol": ["H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar"],
    "Z": range(1, 19),
    "mass": [1.008, 4.003, 6.94, 9.012, 10.81, 12.011, 14.007, 15.999, 18.998, 20.180,
             22.990, 24.305, 26.982, 28.085, 30.974, 32.06, 35.45, 39.948],
    "electronegativity": [2.20, np.nan, 0.98, 1.57, 2.04, 2.55, 3.04, 3.44, 3.98, np.nan,
                          0.93, 1.31, 1.61, 1.90, 2.19, 2.58, 3.16, np.nan],
    "radius_pm": [53, 31, 167, 112, 87, 67, 56, 48, 42, 38, 190, 145, 118, 111, 98, 88, 79, 71],
    "group": [1, 18, 1, 2, 13, 14, 15, 16, 17, 18, 1, 2, 13, 14, 15, 16, 17, 18],
    "period": [1] * 2 + [2] * 8 + [3] * 8,
}).set_index("symbol")
elements.head()
```

```{code-cell} ipython3
elements.info()
elements.describe().round(2)
```

## Selecting data

```{code-cell} ipython3
print(elements.loc["C"])                       # by label
print(elements.iloc[0:3, 0:2])                 # by position
print(elements.loc[["Na", "Cl"], ["mass", "electronegativity"]])
```

### Filtering with conditions

```{code-cell} ipython3
halogens = elements[elements["group"] == 17]
very_en = elements.query("electronegativity > 3")
print(halogens.index.tolist(), very_en.index.tolist())
```

## Adding and transforming columns

```{code-cell} ipython3
elements["block"] = np.where(elements["group"] <= 2, "s", "p")
elements.loc["He", "block"] = "s"
elements["density_proxy"] = elements["mass"] / elements["radius_pm"] ** 3 * 1e6
elements.sort_values("radius_pm", ascending=False).head()
```

## Missing data

```{code-cell} ipython3
print(elements["electronegativity"].isna().sum(), "missing values (noble gases)")
print(elements["electronegativity"].mean().round(3))      # NaNs are skipped automatically
elements.dropna(subset=["electronegativity"]).shape
```

## Grouping and aggregating

```{code-cell} ipython3
elements.groupby("period")[["radius_pm", "electronegativity"]].agg(["mean", "min", "max"]).round(2)
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7, 4))
for period, g in elements.groupby("period"):
    ax.plot(g["group"], g["radius_pm"], "o-", ms=8, label=f"period {period}")
    for sym, r in g.iterrows():
        ax.annotate(sym, (r["group"], r["radius_pm"]), textcoords="offset points", xytext=(0, 7), ha="center", fontsize=8)
ax.set(xlabel="Group", ylabel="Atomic radius / pm", title="Atoms shrink across a period")
ax.legend(); plt.show()
```

## Reading and writing files

```{code-cell} ipython3
elements.to_csv("elements.csv")
df = pd.read_csv("elements.csv", index_col="symbol")
print(df.shape)
# pd.read_excel("lab.xlsx"), pd.read_json(...), pd.read_html(...) work the same way
```

## A real workflow: kinetics lab data

Three student groups measured the decolourisation of crystal violet by NaOH at different temperatures.

```{code-cell} ipython3
rng = np.random.default_rng(7)
records = []
for group, T in [("A", 298), ("B", 308), ("C", 318)]:
    k = 1e9 * np.exp(-60_000 / (8.314 * T))          # pseudo-first-order rate constant
    for t in range(0, 601, 60):
        records.append({"group": group, "T_K": T, "t_s": t,
                        "absorbance": 0.9 * np.exp(-k * t) + rng.normal(0, 0.005)})
kin = pd.DataFrame(records)
kin.sample(5, random_state=1)
```

Fit ln A versus t for each group with `groupby` + `apply`:

```{code-cell} ipython3
def first_order_k(g):
    slope, _ = np.polyfit(g["t_s"], np.log(g["absorbance"]), 1)
    return pd.Series({"k_s-1": -slope})

rates = kin.groupby(["group", "T_K"]).apply(first_order_k, include_groups=False).reset_index()
rates["1/T"] = 1 / rates["T_K"]
rates["ln k"] = np.log(rates["k_s-1"])
slope, intercept = np.polyfit(rates["1/T"], rates["ln k"], 1)
print(rates.round(6))
print(f"Ea = {-slope * 8.314 / 1000:.1f} kJ/mol")
```

## Pivot tables and reshaping

```{code-cell} ipython3
wide = kin.pivot_table(index="t_s", columns="T_K", values="absorbance")
ax = wide.plot(marker="o", colormap="plasma", figsize=(7, 4))
ax.set(ylabel="Absorbance", title="Crystal violet fading at three temperatures")
plt.show()
```

## Merging tables

```{code-cell} ipython3
compounds = pd.DataFrame({"name": ["aspirin", "ibuprofen", "paracetamol", "caffeine"],
                          "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                                     "CC(=O)Nc1ccc(O)cc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]})
assays = pd.DataFrame({"name": ["aspirin", "caffeine", "paracetamol", "naproxen"],
                       "mp_C": [135, 235, 169, 153]})
print(pd.merge(compounds, assays, on="name", how="left"))
```

## pandas meets RDKit

`PandasTools` puts molecule drawings directly into a DataFrame.

```{code-cell} ipython3
from rdkit.Chem import PandasTools, Descriptors
PandasTools.AddMoleculeColumnToFrame(compounds, "smiles", "structure")
compounds["MW"] = compounds["structure"].apply(Descriptors.MolWt).round(2)
compounds
```

## Time series from instruments

```{code-cell} ipython3
idx = pd.date_range("2026-11-27 09:00", periods=240, freq="min")
temp = 25 + 10 * (1 - np.exp(-np.arange(240) / 40)) + rng.normal(0, 0.3, 240)
ts = pd.Series(temp, index=idx, name="T_C")
fig, ax = plt.subplots(figsize=(8, 3.5))
ts.plot(ax=ax, alpha=0.4, label="raw")
ts.rolling(15).mean().plot(ax=ax, lw=2.5, color="#e5487a", label="15-min rolling mean")
ts.resample("30min").mean().plot(ax=ax, marker="s", ls="", color="k", label="30-min average")
ax.set_ylabel("Reactor T / °C"); ax.legend(); plt.show()
```

## Exercises

**★ 1.** From `elements`, list the period-3 elements with electronegativity above 2.

**★★ 2.** Add a column with the number of valence electrons and plot it against electronegativity.

**★★★ 3.** Repeat the crystal-violet analysis, but estimate the uncertainty in Ea by bootstrap resampling the time points 1 000 times.
