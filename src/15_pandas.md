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

Most chemical data is a table: compounds in rows, properties in columns, or time points in rows and sensor readings in columns. NumPy arrays (Chapter 12) are fast, but they hold numbers of one type and identify rows and columns only by position, so you must remember that column 3 is the boiling point. **pandas** is Python's spreadsheet. Its tables have *labelled* rows and columns, each column can have its own type (text, numbers, dates), missing values are handled gracefully, and one line of code can filter, sort, group, reshape or join whole tables. Behind the scenes each column is a NumPy array, so it stays fast. pandas is the tool you will use to load, clean and summarise almost every dataset in the rest of the book, and the format that Seaborn and scikit-learn expect. This chapter builds a small periodic table and a set of kinetics measurements and works through the everyday operations.

pandas is conventionally imported as `pd`:

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Series and DataFrames

pandas has two main data structures. A **Series** is a single labelled column: a one-dimensional array of values, each with a label called its **index**. A **DataFrame** is a table: several Series that share the same index, one per column. You can think of a Series as a dictionary that also behaves like an array, and a DataFrame as a dictionary of such columns.

The simplest Series is made from a list, and pandas labels the values 0, 1, 2, … automatically:

```{code-cell} ipython3
pd.Series([100.0, 78.4, 56.1])
```

A Series made from a dictionary uses the keys as the index, so values can be looked up by name, and methods such as `max()` and `idxmin()` (the label of the smallest value) work on the whole column:

```{code-cell} ipython3
bp = pd.Series({"water": 100.0, "ethanol": 78.4, "acetone": 56.1, "hexane": 68.7}, name="bp_C")
print(bp, "\n")
print(bp["ethanol"], bp.max(), bp.idxmin())
```

A DataFrame is usually built from a dictionary of equal-length lists, one per column. `np.nan` ("not a number") marks a missing value, here the electronegativities of the noble gases, which are not defined on the Pauling scale. `set_index("symbol")` makes the symbol column the row labels, so rows can be found by element. `head()` shows the first five rows, which is the standard first look at any table.

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

`info()` lists each column with its type and number of non-missing values, and `describe()` gives summary statistics (count, mean, standard deviation, quartiles) for every numeric column. Together they tell you at once whether the data loaded as you expected.

```{code-cell} ipython3
elements.info()
elements.describe().round(2)
```

## Selecting data

Selecting means picking out the rows and columns you want. A single column is chosen with square brackets and its name, `df["mass"]`, which returns a Series; a list of names returns a smaller DataFrame. For rows, pandas has two separate tools so that there is never any doubt about what you mean. `.loc[...]` selects by **label** (the index and column names), and `.iloc[...]` selects by integer **position**, like a NumPy array. Both accept `[rows, columns]`.

```{code-cell} ipython3
print(elements["mass"].head(3))              # one column
print(elements[["Z", "mass"]].head(3))       # several columns
```

```{code-cell} ipython3
print(elements.loc["C"])                       # by label
print(elements.iloc[0:3, 0:2])                 # by position
print(elements.loc[["Na", "Cl"], ["mass", "electronegativity"]])
```

### Filtering with conditions

Filtering keeps only the rows that satisfy a condition. A comparison on a column, such as `elements["group"] == 17`, gives a Series of `True` and `False`, one per row, and putting it inside square brackets keeps the `True` rows. This is the same Boolean masking as in NumPy. Conditions combine with `&` (and) and `|` (or), each wrapped in parentheses. `query()` accepts the same condition as a readable string.

```{code-cell} ipython3
print((elements["group"] == 17).head())     # the mask itself
print(elements[(elements["period"] == 2) & (elements["radius_pm"] < 50)].index.tolist())
```

```{code-cell} ipython3
halogens = elements[elements["group"] == 17]
very_en = elements.query("electronegativity > 3")
print(halogens.index.tolist(), very_en.index.tolist())
```

## Adding and transforming columns

A new column is created by assigning to a new name, and arithmetic on columns works element by element, as with NumPy arrays, so `df["a"] / df["b"]` divides row by row. `np.where(condition, x, y)` picks `x` where the condition is true and `y` elsewhere, which turns a test into a category. Individual cells are changed with `.loc[row, column] = value`. `sort_values` reorders the rows by a column.

```{code-cell} ipython3
elements["mass_per_proton"] = elements["mass"] / elements["Z"]
elements[["Z", "mass", "mass_per_proton"]].head(3)
```

Helium sits in group 18 but has an s² configuration, so it is corrected by hand after the automatic s/p assignment. The quantity mass/radius³ is a rough proxy for how densely packed an atom's mass is.

```{code-cell} ipython3
elements["block"] = np.where(elements["group"] <= 2, "s", "p")
elements.loc["He", "block"] = "s"
elements["density_proxy"] = elements["mass"] / elements["radius_pm"] ** 3 * 1e6
elements.sort_values("radius_pm", ascending=False).head()
```

## Missing data

Real datasets have gaps: a measurement failed, a property is undefined, or a value was never recorded. pandas marks missing values as `NaN` and handles them consistently. `isna()` gives a mask of the missing cells, and `.sum()` on that mask counts them (since `True` counts as 1). Statistics such as `mean()` skip missing values automatically, so one gap does not spoil a column. `dropna()` removes rows with missing values and `fillna(value)` replaces them; which to use depends on the science, because filling in a value is a claim about data you do not have.

```{code-cell} ipython3
print(elements["electronegativity"].isna().sum(), "missing values (noble gases)")
print(elements["electronegativity"].mean().round(3))      # NaNs are skipped automatically
elements.dropna(subset=["electronegativity"]).shape
```

## Grouping and aggregating

**Grouping** splits a table into groups that share a value in some column, applies a calculation to each group, and combines the results into a new table. This "split–apply–combine" pattern answers questions such as "what is the mean atomic radius in each period?" without writing any loops. `groupby("period")` does the splitting, and a statistic such as `mean()` or several statistics through `agg([...])` does the rest.

```{code-cell} ipython3
elements.groupby("period")["radius_pm"].mean()
```

```{code-cell} ipython3
elements.groupby("period")[["radius_pm", "electronegativity"]].agg(["mean", "min", "max"]).round(2)
```

Looping over a `groupby` gives each group's value and its sub-table in turn, which is a convenient way to plot one line per group. `iterrows()` walks through the rows of a table one at a time, here to label each point with its symbol.

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

pandas reads and writes most table formats with a matching pair of functions: `pd.read_csv` and `df.to_csv`, `pd.read_excel` and `df.to_excel`, `pd.read_json` and `df.to_json`. `read_csv` works out the column types itself and turns empty cells into `NaN`, so the manual conversion of Chapter 8 is no longer needed. `index_col=` says which column holds the row labels.

```{code-cell} ipython3
elements.to_csv("elements.csv")
df = pd.read_csv("elements.csv", index_col="symbol")
print(df.shape)
# pd.read_excel("lab.xlsx"), pd.read_json(...), pd.read_html(...) work the same way
```

## A real workflow: kinetics lab data

The operations above combine into real analyses. Three student groups measured the decolourisation of crystal violet by NaOH at different temperatures. With NaOH in large excess, the reaction is **pseudo-first-order**: the absorbance decays as $A = A_0 e^{-kt}$, so $\ln A$ falls linearly with time and the slope is $-k$. Repeating at several temperatures and plotting $\ln k$ against $1/T$ gives the activation energy from the Arrhenius equation, $\ln k = \ln A - E_a/RT$. The data are simulated with a known $E_a$ of 60 kJ mol⁻¹ plus noise, stored in *long* format: one row per measurement, with columns saying which group, temperature and time it belongs to.

```{code-cell} ipython3
rng = np.random.default_rng(7)
records = []
for group, T in [("A", 298), ("B", 308), ("C", 318)]:
    k = 3e7 * np.exp(-60_000 / (8.314 * T))          # pseudo-first-order rate constant
    for t in range(0, 601, 60):
        records.append({"group": group, "T_K": T, "t_s": t,
                        "absorbance": 0.9 * np.exp(-k * t) + rng.normal(0, 0.005)})
kin = pd.DataFrame(records)
kin.sample(5, random_state=1)
```

`groupby` plus `apply` runs a function of your own on each group's sub-table. Here `first_order_k` fits a straight line to $\ln A$ against $t$ with `np.polyfit` and returns the rate constant; `reset_index()` turns the group labels back into ordinary columns. A second straight-line fit, of $\ln k$ against $1/T$, then gives $E_a = -\text{slope} \times R$.

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

The recovered activation energy is close to the 60 kJ mol⁻¹ used to simulate the data, which confirms the analysis pipeline.

## Pivot tables and reshaping

Data in long format (one row per measurement) is ideal for grouping, but for plotting or comparing side by side you often want *wide* format, with one column per temperature. A **pivot table** reshapes long data into wide: `index=` becomes the rows, `columns=` supplies the new column names, and `values=` fills the cells. A DataFrame's own `.plot()` method then draws one line per column.

```{code-cell} ipython3
wide = kin.pivot_table(index="t_s", columns="T_K", values="absorbance")
ax = wide.plot(marker="o", colormap="plasma", figsize=(7, 4))
ax.set(ylabel="Absorbance", title="Crystal violet fading at three temperatures")
plt.show()
```

## Merging tables

Information about the same compounds often lives in separate tables: structures in one file, measured melting points in another. **Merging** joins two tables on a shared column, like a database join. `pd.merge(left, right, on="name", how=...)` matches rows with the same name. `how="inner"` keeps only names present in both tables, `how="left"` keeps every row of the left table and fills gaps with `NaN`, and `how="outer"` keeps everything. Here ibuprofen has no melting point, and naproxen, which has no structure, is dropped.

```{code-cell} ipython3
compounds = pd.DataFrame({"name": ["aspirin", "ibuprofen", "paracetamol", "caffeine"],
                          "smiles": ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
                                     "CC(=O)Nc1ccc(O)cc1", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"]})
assays = pd.DataFrame({"name": ["aspirin", "caffeine", "paracetamol", "naproxen"],
                       "mp_C": [135, 235, 169, 153]})
print(pd.merge(compounds, assays, on="name", how="left"))
```

## pandas meets RDKit

A DataFrame column can hold any Python object, including RDKit molecules. RDKit's `PandasTools` module adds a column of molecule objects built from a SMILES column and displays each one as a small drawing in the table. Any function can then be applied to every molecule with `.apply()`, which calls it once per row, here to compute the molecular weight.

```{code-cell} ipython3
from rdkit.Chem import PandasTools, Descriptors
PandasTools.AddMoleculeColumnToFrame(compounds, "smiles", "structure")
compounds["MW"] = compounds["structure"].apply(Descriptors.MolWt).round(2)
compounds
```

## Time series from instruments

Many instruments log a reading every second or minute. When the index of a Series is made of dates and times, pandas treats it as a **time series** and offers time-aware tools. `pd.date_range` creates a sequence of timestamps. `rolling(15).mean()` computes a moving average over a 15-point window, which smooths noise while following the trend. `resample("30min").mean()` groups the readings into 30-minute bins and averages each bin, which reduces a long log to a manageable summary. The reactor temperature below approaches its set point exponentially, with measurement noise.

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

```{admonition} Summary
:class: tip
- A **Series** is a labelled column; a **DataFrame** is a table of Series sharing an index.
- Look at new data with `head()`, `info()` and `describe()`.
- Select columns with `df["col"]`, rows by label with `.loc` and by position with `.iloc`; filter with Boolean masks or `query()`.
- Column arithmetic is element-wise; `np.where` creates categories; `sort_values` sorts.
- Missing values are `NaN`: count with `isna().sum()`, handle with `dropna` or `fillna`.
- `groupby` performs split–apply–combine; `pivot_table` reshapes long to wide; `merge` joins tables.
- `read_csv`/`to_csv` load and save; time-indexed data supports `rolling` and `resample`.
```

## Exercises

**★ 1.** From `elements`, list the period-3 elements with electronegativity above 2.

**★★ 2.** Add a column with the number of valence electrons and plot it against electronegativity.

**★★★ 3.** Repeat the crystal-violet analysis, but estimate the uncertainty in Ea by bootstrap resampling the time points 1 000 times.
