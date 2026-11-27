---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Analytical Chemistry: ML for Spectra

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 27, 31 and 32.
```

Analytical chemists were doing machine learning before it had that name: **chemometrics** has used principal components and partial least squares for decades {cite:p}`wold2001`. Spectra are ideal input for data-driven methods. A single near-infrared (NIR) or IR measurement takes seconds, needs little or no sample preparation and produces hundreds of numbers, one absorbance per wavelength. The problem is that these numbers are strongly correlated and heavily overlapped: in a food sample the water, fat and sugar bands sit on top of one another, so no single wavelength measures any one component. **Calibration** is the task of learning, from samples of known composition, a model that turns a new spectrum into concentrations; **classification** assigns a spectrum to a category, such as the functional group present. Chemometric methods compress the many correlated wavelengths into a few meaningful directions before fitting, which is why they work with far fewer samples than wavelengths. This chapter builds the classical tools, then adds modern classifiers and a shift predictor.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
rng = np.random.default_rng(42)
```

## Simulating a spectral dataset

We build NIR-like spectra of mixtures: three components with overlapping bands, random concentrations, plus baseline drift and noise. This is the classic calibration problem. By the Beer–Lambert law (Chapter 27), the spectrum of a mixture is the concentration-weighted sum of the pure-component spectra, which in matrix form is `C @ pure`: each row of `C` holds one sample's composition and each row of `pure` one component's spectrum. A two-component, three-wavelength example shows the multiplication:

```{code-cell} ipython3
pure_demo = np.array([[1.0, 0.5, 0.0],     # component A at three wavelengths
                      [0.0, 0.5, 1.0]])    # component B
C_demo = np.array([[0.8, 0.2]])            # 80 % A, 20 % B
print(C_demo @ pure_demo)
```

The compositions are drawn from a **Dirichlet** distribution, which produces random fractions that always sum to 1. Real spectra also suffer **baseline drift**, a slowly varying offset caused by light scattering and instrument changes, which is added as a random tilt and offset for each sample.

```{code-cell} ipython3
wl = np.linspace(1100, 2500, 300)                 # nm
def band(c, w, h): return h * np.exp(-0.5 * ((wl - c) / w) ** 2)
pure = np.array([band(1450, 60, 1.0) + band(1930, 80, 0.8),           # water
                 band(1700, 50, 0.9) + band(2300, 70, 0.7),           # oil
                 band(1580, 40, 0.6) + band(2100, 90, 0.5)])          # sugar
names = ["water", "oil", "sugar"]

n = 220
C = rng.dirichlet([2, 2, 2], n)                                        # compositions sum to 1
drift = np.outer(rng.normal(0, 0.05, n), np.linspace(0, 1, wl.size)) + rng.normal(0, 0.02, (n, 1))
S = C @ pure + drift + rng.normal(0, 0.01, (n, wl.size))

fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
for p, nm, c in zip(pure, names, ["#3b82f6", "#f2a93b", "#0f9d8a"]):
    axes[0].plot(wl, p, color=c, lw=2, label=nm)
axes[0].legend(); axes[0].set(xlabel="λ / nm", title="Pure-component spectra")
for s in S[:40]:
    axes[1].plot(wl, s, lw=0.6, alpha=0.7)
axes[1].set(xlabel="λ / nm", title="40 mixture spectra with drift and noise")
plt.tight_layout(); plt.show()
```

## Preprocessing: the chemometrician's toolbox

**Preprocessing** removes variation that has nothing to do with chemistry before any model is fitted. **Standard normal variate** (SNV) subtracts each spectrum's own mean and divides by its own standard deviation, which removes offsets and multiplicative scaling sample by sample. **Multiplicative scatter correction** (MSC) does something similar by fitting each spectrum as a straight-line function of the average spectrum and undoing that line. A **second derivative**, computed with the Savitzky–Golay filter (Chapter 27), removes constant offsets and linear slopes entirely and sharpens overlapping bands, at the cost of amplifying noise. SNV is easy to verify on a single spectrum: afterwards its mean is 0 and its standard deviation 1.

```{code-cell} ipython3
x_demo = np.array([[2.0, 3.0, 5.0, 4.0]])
x_snv = (x_demo - x_demo.mean()) / x_demo.std()
print(x_snv.round(2), x_snv.mean().round(2), x_snv.std().round(2))
```

```{code-cell} ipython3
def snv(X):
    """Standard normal variate: centre and scale each spectrum."""
    return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)

def msc(X):
    """Multiplicative scatter correction against the mean spectrum."""
    ref = X.mean(0)
    out = np.empty_like(X)
    for i, x in enumerate(X):
        a, b = np.polyfit(ref, x, 1)
        out[i] = (x - b) / a
    return out

d2 = savgol_filter(S, 15, 3, deriv=2, axis=1)
variants = {"raw": S, "SNV": snv(S), "MSC": msc(S), "2nd derivative": d2}
fig, axes = plt.subplots(1, 4, figsize=(16, 3.2))
for ax, (k, v) in zip(axes, variants.items()):
    for s in v[:25]:
        ax.plot(wl, s, lw=0.6, alpha=0.7)
    ax.set_title(k); ax.set_xlabel("λ / nm")
plt.tight_layout(); plt.show()
```

## PCA for exploration and outlier detection

PCA (Chapter 33) is the first thing to run on a new spectral dataset. The **scores** are the coordinates of each sample along the principal components, and show how samples group; the **loadings** are the components themselves, spectra-like patterns that show which wavelengths vary together. Here the three compositions sum to 1, so only two directions of chemical variation exist, plus whatever drift survived preprocessing.

PCA also detects **outliers**, samples unlike the calibration set, in two complementary ways. **Hotelling's T²** measures how far a sample lies from the centre *within* the PCA model, so a large value means an extreme but familiar kind of sample. The **Q residual** is the squared part of the spectrum that the model cannot reconstruct, so a large value means something new, such as a contaminant. The dashed lines mark the 95th percentile of each.

```{code-cell} ipython3
from sklearn.decomposition import PCA
pca = PCA(4).fit(snv(S))
T = pca.transform(snv(S))
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
sc = axes[0].scatter(T[:, 0], T[:, 1], c=C[:, 0], cmap="Blues", s=25, edgecolor="grey")
plt.colorbar(sc, ax=axes[0], label="water fraction"); axes[0].set(xlabel="PC1", ylabel="PC2", title="Scores")
for i in range(3):
    axes[1].plot(wl, pca.components_[i], label=f"PC{i+1} ({pca.explained_variance_ratio_[i]:.0%})")
axes[1].legend(); axes[1].set(xlabel="λ / nm", title="Loadings look like pure spectra")
# Hotelling T² vs Q residuals for outlier detection
T2 = (T ** 2 / pca.explained_variance_).sum(1)
Q = ((snv(S) - pca.inverse_transform(T)) ** 2).sum(1)
axes[2].scatter(T2, Q, s=20, color="#6c4ad6")
axes[2].axvline(np.percentile(T2, 95), ls="--", color="#e5487a"); axes[2].axhline(np.percentile(Q, 95), ls="--", color="#e5487a")
axes[2].set(xlabel="Hotelling $T^2$", ylabel="Q residual", title="Outlier map")
plt.tight_layout(); plt.show()
```

## PLS regression: the chemometric workhorse

PLS finds the directions in the spectra that best explain the **concentrations**, not just the variance. **Partial least squares** regression is like PCA followed by linear regression, with one important difference: PCA chooses directions that capture the most variation in the spectra, which may be drift that is irrelevant to concentration, whereas PLS chooses directions (called *latent variables* or components) that are most strongly correlated with the property to be predicted. The number of components is the key setting: too few and the model misses real signal, too many and it fits the noise. It is chosen by cross-validation, from the **root-mean-square error of cross-validation** (RMSECV), at the point where adding components stops helping.

On noise-free data built from two pure spectra, one PLS component per independent component is enough for a near-perfect fit:

```{code-cell} ipython3
from sklearn.cross_decomposition import PLSRegression
c_toy = np.linspace(0, 1, 6)[:, None]
X_toy = c_toy * np.array([1.0, 0.5, 0.0]) + (1 - c_toy) * np.array([0.0, 0.5, 1.0])
print(PLSRegression(1).fit(X_toy, c_toy).score(X_toy, c_toy))
```

```{code-cell} ipython3
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import r2_score, mean_absolute_error

Xp = snv(S)
cv = KFold(5, shuffle=True, random_state=0)
rmsecv = []
for k in range(1, 11):
    p = cross_val_predict(PLSRegression(k), Xp, C, cv=cv)
    rmsecv.append(np.sqrt(((p - C) ** 2).mean()))
best_k = int(np.argmin(rmsecv)) + 1
plt.plot(range(1, 11), rmsecv, "o-", color="#6c4ad6"); plt.axvline(best_k, ls=":", color="k")
plt.xlabel("number of PLS components"); plt.ylabel("RMSECV"); plt.title(f"Choosing the model size (best = {best_k})")
plt.show()

pred = cross_val_predict(PLSRegression(best_k), Xp, C, cv=cv)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
for ax, i, c in zip(axes, range(3), ["#3b82f6", "#f2a93b", "#0f9d8a"]):
    ax.scatter(C[:, i], pred[:, i], s=22, color=c, edgecolor="grey")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set(xlabel=f"true {names[i]} fraction", ylabel="predicted", aspect="equal",
           title=f"{names[i]}: R² = {r2_score(C[:, i], pred[:, i]):.3f}, MAE = {mean_absolute_error(C[:, i], pred[:, i]):.3f}")
plt.tight_layout(); plt.show()
```

### PLS versus other regressors

PLS is not the only option. Ridge regression (Chapter 31) on the full spectrum handles correlated wavelengths by shrinking the coefficients instead of compressing the spectra, and a random forest can capture non-linear effects. Comparing them on the same folds shows whether the extra flexibility is worth anything.

```{code-cell} ipython3
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
models = {"PLS": PLSRegression(best_k), "ridge (full spectrum)": make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 20))),
          "random forest": RandomForestRegressor(200, n_jobs=-1, random_state=0),
          "PCA + ridge": make_pipeline(PCA(best_k), RidgeCV(alphas=np.logspace(-3, 3, 20)))}
rows = []
for name, m in models.items():
    p = cross_val_predict(m, Xp, C[:, 1], cv=cv)
    rows.append({"model": name, "R² (oil)": r2_score(C[:, 1], p), "MAE": mean_absolute_error(C[:, 1], p)})
pd.DataFrame(rows).set_index("model").round(3)
```

The three linear models perform almost identically, as expected when the spectra follow the Beer–Lambert law. The random forest does better here, probably because SNV divides each spectrum by its own standard deviation, which makes the link between the preprocessed spectrum and the concentration slightly non-linear. A different preprocessing choice could reverse the ranking, which is why the comparison is worth running on every new dataset.

## Classification: identifying a functional group from IR

Each simulated IR spectrum is built from the characteristic bands of its functional groups (Chapter 27), with random shifts in position, width and height so that no two are identical. The task is to name the compound class from the spectrum. A **support vector machine** (SVM) is a classifier that finds the boundaries separating the classes with the widest possible margin, and with its default kernel it can draw curved boundaries. The **classification report** lists, for each class, the *precision* (of the spectra predicted to be in the class, the fraction that really are) and the *recall* (of the spectra really in the class, the fraction found); the **confusion matrix** shows which classes are mistaken for which. `stratify=yir` keeps the class proportions the same in the training and test sets.

```{code-cell} ipython3
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split

wn = np.linspace(4000, 500, 400)
BANDS = {"OH": (3350, 130, 0.7), "NH": (3320, 60, 0.5), "CH": (2950, 35, 0.6), "C=O": (1715, 20, 1.0),
         "C=C": (1640, 22, 0.35), "CN": (2240, 15, 0.5), "C-O": (1150, 40, 0.6), "NO2": (1530, 25, 0.8)}
CLASSES = {"alcohol": ["OH", "CH", "C-O"], "ketone": ["CH", "C=O"], "carboxylic acid": ["OH", "CH", "C=O", "C-O"],
           "amine": ["NH", "CH"], "amide": ["NH", "CH", "C=O"], "nitrile": ["CH", "CN"],
           "nitro": ["CH", "NO2"], "alkene": ["CH", "C=C"]}

def make_spectrum(groups, r):
    y = np.zeros(wn.size)
    for g in groups:
        c, w, h = BANDS[g]
        y += h * r.uniform(0.7, 1.3) * np.exp(-0.5 * ((wn - c - r.normal(0, 12)) / (w * r.uniform(0.8, 1.3))) ** 2)
    return y + r.normal(0, 0.03, wn.size) + r.uniform(0, 0.05)

r = np.random.default_rng(1)
Xir = np.array([make_spectrum(g, r) for _ in range(120) for g in CLASSES.values()])
yir = np.array([k for _ in range(120) for k in CLASSES])
Xir = snv(savgol_filter(Xir, 11, 3, axis=1))
Xtr, Xte, ytr, yte = train_test_split(Xir, yir, test_size=0.25, stratify=yir, random_state=0)
svc = make_pipeline(StandardScaler(), SVC(C=5, probability=True)).fit(Xtr, ytr)
print(classification_report(yte, svc.predict(Xte)))
fig, ax = plt.subplots(figsize=(6.5, 5.5))
ConfusionMatrixDisplay.from_estimator(svc, Xte, yte, cmap="Purples", xticks_rotation=60, ax=ax, colorbar=False)
plt.title("IR functional-group classifier"); plt.tight_layout(); plt.show()
```

### Which wavenumbers did it use?

**Permutation importance** works for any model: shuffle the values of one feature across the test samples, which destroys its information, and measure how much the accuracy drops. Wavenumbers whose shuffling hurts most are the ones the model relies on.

```{code-cell} ipython3
from sklearn.inspection import permutation_importance
imp = permutation_importance(svc, Xte, yte, n_repeats=3, random_state=0, n_jobs=-1).importances_mean
plt.figure(figsize=(9, 3.2))
plt.plot(wn, imp, color="#6c4ad6")
for g, (c, _, _) in BANDS.items():
    plt.axvline(c, ls=":", color="grey"); plt.text(c, imp.max() * 1.02, g, ha="center", fontsize=8, rotation=90)
plt.gca().invert_xaxis(); plt.xlabel("wavenumber / cm$^{-1}$"); plt.ylabel("permutation importance")
plt.title("The model learned the real diagnostic bands"); plt.show()
```

The peaks of importance land on the bands a chemist would use. Checking that a model attends to physically meaningful features is the cheapest sanity test there is.

## NMR shift prediction

Replace Chapter 27's hand-written rules with a model trained on atom environments. We generate training data from an additive rule plus noise, then let the model rediscover it from fingerprints. The key change is that the prediction is made for each *atom*, not for the whole molecule. Each hydrogen-bearing carbon is described by an **atom-centred** fingerprint, a Morgan count fingerprint (Chapter 21) built only from the environments around that atom (`fromAtoms=[idx]`), so carbons in similar surroundings get similar vectors and similar shifts. The training labels come from a table of SMARTS patterns (Chapter 21) with typical ¹H shifts; when several patterns match an atom, the later, more specific one overrides the earlier one. A single pattern shows how atoms are matched:

```{code-cell} ipython3
from rdkit import Chem
print(Chem.MolFromSmiles("CCO").GetSubstructMatches(Chem.MolFromSmarts("[CX4;H1,H2,H3][OX2]")))
```

The match contains the carbon and the oxygen; its first atom, atom 1 (the CH₂ next to oxygen), is the one labelled, and it would be labelled with a shift near 3.6 ppm.

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.ensemble import GradientBoostingRegressor

ENV_RULES = [("[CX4H3]", 0.9), ("[CX4H2]", 1.3), ("[CX4H1]", 1.6), ("[CX4;H1,H2,H3][CX3]=O", 2.2),
             ("[CX4;H1,H2,H3]c", 2.4), ("[CX4;H1,H2,H3][OX2]", 3.6), ("[CX4;H1,H2,H3][NX3]", 2.7),
             ("[cH]", 7.25), ("[cH]c[NX3]", 6.6), ("[cH]c[CX3]=O", 7.9), ("[CX3H1]=O", 9.7)]

def labelled_shifts(smi, r):
    m = Chem.MolFromSmiles(smi)
    out = {}
    for smarts, shift in ENV_RULES:
        for match in m.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
            out[match[0]] = shift + r.normal(0, 0.12)
    return m, out

TRAIN_SMILES = ["CCO", "CCCO", "CC(=O)C", "CC(=O)O", "CCOC(C)=O", "Cc1ccccc1", "c1ccccc1", "O=Cc1ccccc1",
                "CCN", "CC(C)O", "COC", "CC(=O)N", "Nc1ccccc1", "O=C(C)c1ccccc1", "CCCC", "CC(C)(C)O",
                "OCc1ccccc1", "CCC(=O)OC", "c1ccncc1", "CN(C)C=O"]
gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=1024, includeChirality=False)

def atom_fp(m, idx):
    return gen.GetCountFingerprintAsNumPy(m, fromAtoms=[idx])

r = np.random.default_rng(0)
Xs, ys = [], []
for smi in TRAIN_SMILES:
    m, shifts = labelled_shifts(smi, r)
    for idx, sh in shifts.items():
        Xs.append(atom_fp(m, idx)); ys.append(sh)
Xs, ys = np.array(Xs), np.array(ys)
gb = GradientBoostingRegressor(random_state=0).fit(Xs, ys)
print(len(ys), "labelled atoms")

test = ["CCOCC", "CC(=O)OC", "Cc1ccc(C)cc1", "CCC(=O)O"]
for smi in test:
    m = Chem.MolFromSmiles(smi)
    pred = {a.GetIdx(): round(float(gb.predict(atom_fp(m, a.GetIdx())[None])[0]), 2)
            for a in m.GetAtoms() if a.GetTotalNumHs() > 0}
    print(f"{smi:14s} predicted δ: {pred}")
```

Most predictions sit close to the rule values, but two show the limits of a small training set. The CH₂ of propanoic acid is predicted at about 3.1 ppm instead of about 2.2, because no training molecule has a CH₂ between a methyl group and a carboxylic acid, and the acid's OH hydrogen receives a meaningless number because no rule ever labelled an OH. Both are applicability-domain failures (Chapter 36). Research-grade predictors use graph networks trained on tens of thousands of assigned spectra, but the workflow (atom environment in, shift out) is exactly this.

```{admonition} Summary
:class: tip
- Mixture spectra are concentration-weighted sums of pure spectra (Beer–Lambert), so calibration is a linear problem.
- Preprocess first: SNV and MSC remove scatter and offsets; second derivatives remove baselines.
- PCA scores, loadings, Hotelling T² and Q residuals explore data and flag outliers.
- PLS regression picks directions correlated with concentration; choose the number of components by RMSECV.
- SVM classifiers identify functional groups; permutation importance checks that they use real bands.
- Atom-centred fingerprints turn NMR shift prediction into a regression per atom.
```

## Exercises

**★ 1.** Add a fourth component to the NIR mixture and check how many PLS components are needed.

**★★ 2.** Corrupt 10 % of the IR spectra with a strong sloping baseline and compare SNV with second derivatives for the classifier.

**★★★ 3.** Train the shift predictor on aliphatic molecules only and test it on aromatics. Explain the failure in terms of the applicability domain (Chapter 36).
