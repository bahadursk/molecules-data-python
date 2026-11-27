# Appendix A: Python and RDKit Cheat Sheets

One-page reminders of the syntax used most often in this book. Each table points to the chapter where the idea is taught properly.

## Python essentials (Chapters 1–11)

| Task | Code |
|---|---|
| Arithmetic, powers, scientific notation | `6.022e23 * 0.5`, `2 ** 10`, `10 ** -2.75` |
| Integer and floor division, remainder | `7 // 2`, `7 % 2` |
| Formatted output | `f"{k:.3e} s⁻¹"`, `f"{T:6.1f} K"`, `f"{x:.2%}"` |
| Strings | `s.upper()`, `s.split(",")`, `"Na" in s`, `s.replace("Cl", "Br")` |
| Lists | `xs.append(1)`, `xs[1:4]`, `xs[-1]`, `sorted(xs, key=len)` |
| Dictionaries | `d = {"H": 1.008}`, `d.get("He", 0)`, `for k, v in d.items():` |
| Sets | `a & b` (intersection), `a.union(b)`, `a - b` (difference) |
| Comprehensions | `[m * 2 for m in masses if m > 10]`, `{s: z for z, s in enumerate(symbols, 1)}` |
| Conditionals and loops | `if ... elif ... else`, `for x in xs:`, `while c > 1e-3:`, `break`, `continue` |
| Functions | `def f(x, n=2, *args, **kw): return ...`, `lambda x: x ** 2` |
| Errors | `try: ... except ValueError as e: ... finally: ...`, `raise ValueError("msg")` |
| Files | `with open("x.csv") as fh: lines = fh.readlines()`, `pathlib.Path("data") / "x.csv"` |
| Classes | `class Molecule:` / `def __init__(self, ...)`, `@property`, `@classmethod`, `__repr__` |
| Modules | `import numpy as np`, `from math import exp, log`, `pip install rdkit` |

## NumPy (Chapter 12)

| Task | Code |
|---|---|
| Create | `np.array([1, 2])`, `np.zeros((3, 3))`, `np.linspace(0, 1, 101)`, `np.arange(0, 10, 0.5)` |
| Shape | `a.shape`, `a.reshape(3, -1)`, `a.T`, `a[:, None]` (add an axis) |
| Mask | `a[a > 0]`, `np.where(a > 0, a, 0)` |
| Reduce | `a.sum(axis=0)`, `a.mean()`, `a.std(ddof=1)`, `np.argmax(a)` |
| Linear algebra | `A @ x`, `np.linalg.solve(A, b)`, `np.linalg.eigh(H)`, `np.linalg.svd(M)`, `np.linalg.lstsq(X, y)` |
| Random | `rng = np.random.default_rng(0)`, `rng.normal(0, 1, 100)`, `rng.choice(xs, 5)` |
| Range | `np.ptp(a)` (the method `a.ptp()` was removed in NumPy 2) |

## pandas (Chapter 15)

| Task | Code |
|---|---|
| Read | `pd.read_csv("x.csv")`; for chemical formulas add `keep_default_na=False, na_values=[""]` |
| Look | `df.head()`, `df.info()`, `df.describe()`, `df["col"].value_counts()` |
| Select | `df["MW"]`, `df[["MW", "logP"]]`, `df.loc[df.MW < 500]`, `df.query("logP > 2 and HBD <= 5")` |
| New column | `df["MW_kDa"] = df.MW / 1000`, `df.assign(ratio=lambda d: d.a / d.b)` |
| Group | `df.groupby("scaffold").logS_sim.agg(["mean", "count"])` |
| Combine | `pd.merge(a, b, on="smiles")`, `pd.concat([a, b])` |
| Reshape | `df.pivot_table(index="ligand", columns="base", values="yield_sim")`, `df.melt(...)` |
| Missing values | `df.isna().sum()`, `df.dropna()`, `df.fillna(0)` |

## Matplotlib, Seaborn and Plotly (Chapters 13–14)

| Task | Code |
|---|---|
| Figure and axes | `fig, ax = plt.subplots(1, 2, figsize=(10, 4))` |
| Plot types | `ax.plot`, `ax.scatter(x, y, c=z, cmap="viridis")`, `ax.hist`, `ax.bar`, `ax.imshow`, `ax.contourf` |
| Labels | `ax.set(xlabel="T / K", ylabel="k / s$^{-1}$", title="...")` |
| Spectroscopy convention | `ax.invert_xaxis()` (IR and NMR) |
| Colour bar | `fig.colorbar(sc, ax=ax, label="logS")` |
| Save | `fig.savefig("fig.png", dpi=300, bbox_inches="tight")` |
| Seaborn | `sns.scatterplot(data=df, x="MW", y="logP", hue="toxic_sim")`, `sns.heatmap(df.corr(), annot=True)` |
| Plotly | `px.scatter(df, x="MW", y="logP", hover_name="smiles")` |

## SciPy, SymPy, pint and uncertainties (Chapters 16–18)

| Task | Code |
|---|---|
| Curve fit | `p, cov = curve_fit(f, x, y, p0=[...])`; errors `np.sqrt(np.diag(cov))` |
| Root | `optimize.brentq(f, a, b)` |
| Minimise | `optimize.minimize(f, x0)` |
| ODE | `solve_ivp(rhs, (0, t_end), y0, t_eval=t)` |
| Peaks | `signal.find_peaks(y, prominence=0.1)` |
| Integrate | `integrate.quad(f, 0, np.inf)`, `np.trapezoid(y, x)` |
| SymPy | `x = sp.symbols("x")`, `sp.solve(eq, x)`, `sp.diff(f, x)`, `sp.integrate(f, (x, 0, sp.oo))` |
| Units | `ureg = pint.UnitRegistry()`, `(5 * ureg.kJ / ureg.mol).to("kcal/mol")` |
| Temperatures | `Q_(25.0, ureg.degC).to("K")` (offset units need `Q_`) |
| Uncertainty | `from uncertainties import ufloat`, `c = ufloat(0.1000, 0.0002)` |

## RDKit (Chapters 19–24)

| Task | Code |
|---|---|
| Parse | `m = Chem.MolFromSmiles("CCO")` (returns `None` if invalid) |
| Write | `Chem.MolToSmiles(m)`, `Chem.MolToInchiKey(m)`, `Chem.MolToMolBlock(m)` |
| Draw | `Draw.MolToImage(m)`, `Draw.MolsToGridImage(ms, molsPerRow=4, legends=names)` |
| Hydrogens | `Chem.AddHs(m)`, `Chem.RemoveHs(m)` |
| Descriptors | `Descriptors.MolWt(m)`, `Crippen.MolLogP(m)`, `rdMolDescriptors.CalcTPSA(m)` |
| Substructure | `patt = Chem.MolFromSmarts("C(=O)[OH]")`, `m.HasSubstructMatch(patt)`, `m.GetSubstructMatches(patt)` |
| Fingerprint | `gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)`, `fp = gen.GetFingerprint(m)` |
| Similarity | `DataStructs.TanimotoSimilarity(fp1, fp2)`, `DataStructs.BulkTanimotoSimilarity(fp, fps)` |
| Scaffold | `MurckoScaffold.MurckoScaffoldSmiles(mol=m)` |
| Reaction | `rxn = AllChem.ReactionFromSmarts("[C:1](=O)O.[N:2]>>[C:1](=O)[N:2]")`, `rxn.RunReactants((a, b))` |
| 3-D | `mh = Chem.AddHs(m)`, `AllChem.EmbedMolecule(mh, randomSeed=0)`, `AllChem.MMFFOptimizeMolecule(mh)` |
| Atoms and bonds | `for a in m.GetAtoms(): a.GetSymbol(), a.GetDegree(), a.GetIsAromatic()` |

### Useful SMARTS

| Group | SMARTS |
|---|---|
| carboxylic acid | `C(=O)[OH]` |
| ester | `[CX3](=O)[OX2][#6]` |
| primary amine | `[NX3;H2][CX4]` |
| amide | `C(=O)N` |
| nitro | `[N+](=O)[O-]` |
| aryl halide | `c[F,Cl,Br,I]` |
| any aromatic ring atom | `a` |
| ring atom | `[R]` |

## scikit-learn (Chapters 31–36)

| Task | Code |
|---|---|
| Split | `train_test_split(X, y, test_size=0.2, random_state=0)`; for chemistry prefer `GroupKFold` on scaffolds |
| Pipeline | `make_pipeline(StandardScaler(), Ridge(alpha=1.0))` |
| Fit and predict | `model.fit(Xtr, ytr)`, `model.predict(Xte)`, `model.predict_proba(Xte)[:, 1]` |
| Cross-validate | `cross_val_score(model, X, y, cv=GroupKFold(5), groups=scaffolds)` |
| Tune | `GridSearchCV(model, {"alpha": [0.1, 1, 10]}, cv=5)` |
| Metrics | `mean_absolute_error`, `r2_score`, `roc_auc_score`, `confusion_matrix`, `brier_score_loss` |
| Explain | `permutation_importance(model, Xte, yte)`, `PartialDependenceDisplay.from_estimator(...)` |
| Uncertainty | `GaussianProcessRegressor(...).predict(X, return_std=True)`; tree spread across `model.estimators_` |
| Dimensionality | `PCA(n_components=2)`, `TSNE`, `PLSRegression(n_components=3)` |

## PyTorch (Chapters 34, 45, 49)

```python
import torch, torch.nn as nn

model = nn.Sequential(nn.Linear(n_in, 64), nn.ReLU(), nn.Linear(64, 1))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.MSELoss()

for epoch in range(200):
    model.train()
    opt.zero_grad()
    loss = loss_fn(model(Xtr), ytr)
    loss.backward()
    opt.step()

model.eval()
with torch.no_grad():
    pred = model(Xte)
```

## The ML checklist (Chapters 31, 36 and 51)

1. **What is the question?** Write down the decision the model will inform.
2. **Where does the data come from?** Measured, computed or simulated? Units? Duplicates?
3. **How will you split?** Random splits flatter; use scaffold, group or time splits.
4. **What is the baseline?** Predict the mean, or a one-descriptor linear model.
5. **What is the error on genuinely new inputs?** Report it, not the random-split number.
6. **How sure is it?** Ensembles or GP; check calibration.
7. **When should it refuse?** Define the applicability domain.
8. **Can someone else reproduce it?** Seeds, pinned versions, data and code with a DOI.
