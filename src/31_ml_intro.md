---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Machine Learning with scikit-learn

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Parts I–II and Chapter 20.
```

Machine learning (ML) finds patterns in data and uses them to make predictions. In chemistry, that might mean predicting solubility from structure, yield from conditions, or band gap from composition. This chapter teaches the complete workflow with **scikit-learn** {cite:p}`pedregosa2011`, and, above all, how to tell whether a model is any good.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name):
    """Read a book dataset from the local repository, or from GitHub when running in Colab."""
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name)

df = load_data("molecules_sim.csv")
df.head()
```

```{admonition} About the data
:class: warning
`molecules_sim.csv` contains 2 600+ real chemical structures, but their solubility (`logS_sim`), activity (`pIC50_sim`) and toxicity (`toxic_sim`) values are **simulated** from transparent formulas plus noise (see `tools/make_datasets.py`). The dataset behaves like real data for learning, carries no licence restrictions, and lets us know the "true" answer. Never use these values as measurements.
```

## The ML workflow

1. **Frame** the question: what is predicted (target *y*) from what (features *X*)?
2. **Featurise**: turn molecules into numbers.
3. **Split** data into training and test sets.
4. **Train** a model on the training set.
5. **Evaluate** on data the model has never seen.
6. **Interpret** and iterate.

## Features and target

```{code-cell} ipython3
features = ["MW", "logP", "TPSA", "HBD", "RotB"]
X = df[features].values
y = df["logS_sim"].values
print(X.shape, y.shape)
sns.pairplot(df.sample(400, random_state=0), vars=features + ["logS_sim"], corner=True, height=1.5,
             plot_kws=dict(s=8, alpha=0.5, color="#6c4ad6"))
plt.show()
```

## Train / test split

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train), "training,", len(X_test), "test molecules")
```

```{warning}
The test set is a sealed envelope. Look at it **once**, at the very end. If you tune your model while watching test scores, those scores will be over-optimistic.
```

## A first model: linear regression

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

lin = LinearRegression().fit(X_train, y_train)
pred = lin.predict(X_test)

def report(y_true, y_pred, name=""):
    print(f"{name:18s} MAE = {mean_absolute_error(y_true, y_pred):.3f}  RMSE = {mean_squared_error(y_true, y_pred) ** 0.5:.3f}"
          f"  R² = {r2_score(y_true, y_pred):.3f}")

report(y_test, pred, "linear")
print(dict(zip(features, lin.coef_.round(4))))
```

```{code-cell} ipython3
def parity(y_true, y_pred, title, ax=None, color="#6c4ad6"):
    ax = ax or plt.gca()
    ax.scatter(y_true, y_pred, s=10, alpha=0.6, color=color)
    lims = [min(y_true.min(), y_pred.min()) - 0.3, max(y_true.max(), y_pred.max()) + 0.3]
    ax.plot(lims, lims, "k--", lw=1)
    ax.set(xlim=lims, ylim=lims, xlabel="true logS", ylabel="predicted logS",
           title=f"{title}  (R² = {r2_score(y_true, y_pred):.2f})")
    ax.set_aspect("equal")

parity(y_test, pred, "Linear regression"); plt.show()
```

## Preprocessing and pipelines

Many models need features on similar scales. A **pipeline** bundles preprocessing and model, so the same steps are applied identically to training and test data.

```{code-cell} ipython3
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Ridge

poly = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), Ridge(alpha=1.0)).fit(X_train, y_train)
report(y_test, poly.predict(X_test), "quadratic ridge")
```

## Non-linear models

```{code-cell} ipython3
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR

models = {
    "k-nearest neighbours": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7)),
    "support vector": make_pipeline(StandardScaler(), SVR(C=3.0)),
    "random forest": RandomForestRegressor(n_estimators=300, random_state=0, n_jobs=-1),
    "gradient boosting": GradientBoostingRegressor(random_state=0),
}
fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (name, m), c in zip(axes, models.items(), ["#e5487a", "#f2a93b", "#0f9d8a", "#3b82f6"]):
    m.fit(X_train, y_train)
    report(y_test, m.predict(X_test), name)
    parity(y_test, m.predict(X_test), name, ax=ax, color=c)
plt.tight_layout(); plt.show()
```

## Overfitting and the bias–variance trade-off

```{code-cell} ipython3
from sklearn.tree import DecisionTreeRegressor
depths = range(1, 21)
train_err, test_err = [], []
for d in depths:
    t = DecisionTreeRegressor(max_depth=d, random_state=0).fit(X_train, y_train)
    train_err.append(mean_absolute_error(y_train, t.predict(X_train)))
    test_err.append(mean_absolute_error(y_test, t.predict(X_test)))
plt.plot(depths, train_err, "o-", color="#0f9d8a", label="training error")
plt.plot(depths, test_err, "s-", color="#e5487a", label="test error")
best = depths[int(np.argmin(test_err))]
plt.axvline(best, ls=":", color="k"); plt.text(best + 0.3, max(test_err) * 0.95, "sweet spot")
plt.axhline(0.35 * np.sqrt(2 / np.pi), ls="--", color="grey", label="noise floor of the data")
plt.xlabel("tree depth (model complexity)"); plt.ylabel("MAE"); plt.legend()
plt.title("Underfitting ← → overfitting"); plt.show()
```

A model that memorises its training set (deep trees, right) is useless on new molecules. Note also the **noise floor**: no model can beat the experimental (here, simulated) noise.

## Cross-validation and hyperparameter tuning

```{code-cell} ipython3
from sklearn.model_selection import cross_val_score, GridSearchCV, KFold
cv = KFold(5, shuffle=True, random_state=0)
scores = cross_val_score(RandomForestRegressor(n_estimators=200, random_state=0, n_jobs=-1), X_train, y_train, cv=cv,
                         scoring="neg_mean_absolute_error")
print(f"5-fold CV MAE = {-scores.mean():.3f} ± {scores.std():.3f}")

grid = GridSearchCV(make_pipeline(StandardScaler(), SVR()), {"svr__C": [0.3, 1, 3, 10], "svr__gamma": ["scale", 0.05, 0.2]},
                    cv=cv, scoring="neg_mean_absolute_error").fit(X_train, y_train)
print("best:", grid.best_params_, f"CV MAE = {-grid.best_score_:.3f}")
report(y_test, grid.predict(X_test), "tuned SVR (test)")
```

## Classification: predicting toxicity

```{code-cell} ipython3
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay

yc = df["toxic_sim"].values
Xc_train, Xc_test, yc_train, yc_test = train_test_split(X, yc, test_size=0.25, stratify=yc, random_state=1)
from sklearn.ensemble import RandomForestClassifier
rf_c = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0).fit(Xc_train, yc_train)
lr_c = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced")).fit(Xc_train, yc_train)
print(classification_report(yc_test, rf_c.predict(Xc_test), target_names=["non-toxic", "toxic"]))

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
ConfusionMatrixDisplay.from_estimator(rf_c, Xc_test, yc_test, display_labels=["non-toxic", "toxic"], cmap="Purples", ax=axes[0])
RocCurveDisplay.from_estimator(rf_c, Xc_test, yc_test, ax=axes[1], name="random forest", curve_kwargs={"color": "#0f9d8a"})
RocCurveDisplay.from_estimator(lr_c, Xc_test, yc_test, ax=axes[1], name="logistic", curve_kwargs={"color": "#e5487a"})
axes[1].plot([0, 1], [0, 1], "k:")
plt.tight_layout(); plt.show()
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Only about 10 % of the molecules are "toxic". A model that always answers "non-toxic" is 90 % accurate and completely useless. For imbalanced problems report precision, recall, ROC-AUC or balanced accuracy, never accuracy alone.
```

The descriptors used here cannot see the toxic **substructures** (nitro groups, anilines…) that actually drive the label, which is why performance is modest. Chapter 32 fixes this with fingerprints.

## Saving a model

```{code-cell} ipython3
import joblib
joblib.dump(grid.best_estimator_, "solubility_svr.joblib")
reloaded = joblib.load("solubility_svr.joblib")
print(reloaded.predict([[180.2, 1.3, 63.6, 1, 3]]))     # aspirin-like descriptors
```

## Exercises

**★ 1.** Retrain the linear model using only `logP`. How much worse is it?

**★★ 2.** Plot a learning curve (`sklearn.model_selection.learning_curve`) for the random forest. Would more data help?

**★★★ 3.** Predict `pIC50_sim` with the same descriptors. Which model is best, and why do all of them struggle? (Hint: look at how the value was simulated.)
