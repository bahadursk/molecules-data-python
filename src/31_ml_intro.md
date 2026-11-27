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

Machine learning (ML) finds patterns in data and uses them to make predictions. In chemistry, that might mean predicting solubility from structure, yield from conditions, or band gap from composition. Instead of writing the rule yourself, as in Chapter 20's rule of five, you give the computer many examples where the answer is known and let an algorithm *learn* a rule that maps inputs to outputs. This is called **supervised learning**: the training examples come with labels. Predicting a number (solubility) is **regression**; predicting a category (toxic or not) is **classification**. The value of ML is speed and scale: once trained, a model predicts in microseconds for molecules that have never been made. Its danger is that a model can look excellent on the data it has seen and be useless on anything new. This chapter teaches the complete workflow with **scikit-learn** {cite:p}`pedregosa2011`, and, above all, how to tell whether a model is any good.

scikit-learn gives every model the same interface: create the model, call `.fit(X, y)` to train it, then `.predict(X_new)` to use it. The smallest possible example learns a straight line from three points:

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression
X_tiny = [[1], [2], [3]]          # one feature per example, so each row is a list
y_tiny = [2.1, 3.9, 6.0]
tiny = LinearRegression().fit(X_tiny, y_tiny)
print(tiny.coef_, tiny.intercept_, tiny.predict([[4]]))
```

`X` is always two-dimensional (rows are examples, columns are features) and `y` one-dimensional. With that pattern in hand, we load the book's dataset of molecules with simulated properties.

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

Every ML project, from a student exercise to a published drug-discovery model, follows the same six steps. Skipping any of them, especially the separation of training and test data, is the most common cause of results that do not hold up.

1. **Frame** the question: what is predicted (target *y*) from what (features *X*)?
2. **Featurise**: turn molecules into numbers.
3. **Split** data into training and test sets.
4. **Train** a model on the training set.
5. **Evaluate** on data the model has never seen.
6. **Interpret** and iterate.

## Features and target

The **target** $y$ is the quantity you want to predict, here the simulated aqueous solubility logS (log₁₀ of the solubility in mol L⁻¹). The **features** $X$ are the inputs the model may use, here five RDKit descriptors from Chapter 20: molecular weight, logP, polar surface area, hydrogen-bond donors and rotatable bonds. Features must be available for new molecules *before* the target is measured, otherwise the model is useless in practice. A pair plot is a sensible first look: logS falls clearly with logP, as chemistry predicts, and less clearly with size.

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

A model must be judged on molecules it did not learn from, because the goal is to predict *new* molecules. The standard approach is to hold back part of the data as a **test set** before training begins. `train_test_split` shuffles the rows and sets aside a chosen fraction, here 20 %; `random_state` fixes the shuffle so the split is reproducible.

```{code-cell} ipython3
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(X_train), "training,", len(X_test), "test molecules")
```

```{warning}
The test set is a sealed envelope. Look at it **once**, at the very end. If you tune your model while watching test scores, those scores will be over-optimistic.
```

## A first model: linear regression

**Linear regression** predicts the target as a weighted sum of the features plus a constant, $\hat{y} = w_1x_1 + w_2x_2 + \dots + b$, and chooses the weights that minimise the squared errors on the training set. It is fast, easy to interpret (each weight says how much $y$ changes per unit of a feature) and a good **baseline**: any more complex model must beat it to be worth using.

Three numbers summarise how good regression predictions are. The **MAE** (mean absolute error) is the average size of the error, in the units of the target. The **RMSE** (root-mean-square error) is similar but punishes large errors more. **R²** is the fraction of the variance in the target that the model explains: 1 is perfect and 0 is no better than always predicting the mean. A short example makes the definitions concrete:

```{code-cell} ipython3
true_demo = np.array([-2.0, -3.0, -1.0])
pred_demo = np.array([-2.5, -2.8, -1.1])
print("MAE  =", np.mean(np.abs(true_demo - pred_demo)).round(3))
print("RMSE =", np.sqrt(np.mean((true_demo - pred_demo) ** 2)).round(3))
```

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

The negative weight on logP says that oilier molecules are less soluble, as expected. A **parity plot** shows predicted against true values; perfect predictions would lie on the diagonal, and the scatter around it shows the errors at a glance.

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

Molecular weights are in the hundreds while hydrogen-bond donor counts are small integers. Many models need features on similar scales, because they measure distances between examples or penalise large weights, and a feature with big numbers would otherwise dominate. `StandardScaler` rescales each feature to mean 0 and standard deviation 1. Crucially, the scaling parameters must be learned from the *training* set only and then applied unchanged to the test set; computing them from all the data would let information about the test set leak into training.

```{code-cell} ipython3
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler().fit(X_train)
print(scaler.transform(X_train).mean(axis=0).round(2), scaler.transform(X_train).std(axis=0).round(2))
```

Many models need features on similar scales. A **pipeline** bundles preprocessing and model, so the same steps are applied identically to training and test data.

```{code-cell} ipython3
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge

poly = make_pipeline(StandardScaler(), PolynomialFeatures(degree=2), Ridge(alpha=1.0)).fit(X_train, y_train)
report(y_test, poly.predict(X_test), "quadratic ridge")
```

`PolynomialFeatures(degree=2)` adds squares and products of the features, so a linear model can fit curved relationships. **Ridge** regression is linear regression with a penalty on large weights (strength `alpha`), which stops the many extra terms from fitting noise.

## Non-linear models

Real structure–property relationships are rarely straight lines, and several model families capture curves and interactions automatically. **k-nearest neighbours** predicts the average target of the *k* most similar training molecules, a direct use of the similarity principle. **Support vector regression** fits a smooth function through a kernel that measures similarity between examples. A **decision tree** splits the data by a sequence of yes/no questions on the features ("is logP > 2.3?"), and a **random forest** averages hundreds of trees, each trained on a random sample of the data, which makes it robust and hard to overfit. **Gradient boosting** builds trees one after another, each correcting the errors of the ones before. All share the same `fit`/`predict` interface, so comparing them is a loop.

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

A model that is too simple misses real patterns and makes large errors on both training and test data (**underfitting**, or high *bias*). A model that is too flexible fits the noise in the training data as if it were signal: its training error becomes tiny while its test error grows (**overfitting**, or high *variance*). The best model lies between. A decision tree makes this visible, because its depth directly controls its flexibility: a depth-1 tree asks one question, and a depth-20 tree can carve out a region for almost every training molecule.

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

Settings such as tree depth, the number of neighbours or the SVR's `C` are **hyperparameters**: they are chosen before training rather than learned. Choosing them by watching the test score would contaminate the test set, so they are chosen with **cross-validation** on the training set alone. In *k*-fold cross-validation the training data are split into *k* parts (folds); the model is trained on *k* − 1 of them and scored on the remaining one, and this is repeated so that every fold is used once for scoring. The average score estimates performance on new data, and its spread shows how stable that estimate is. `GridSearchCV` runs cross-validation for every combination of hyperparameters in a grid and keeps the best. scikit-learn reports errors as negative numbers (`neg_mean_absolute_error`) because it always maximises scores, so the sign is flipped for printing.

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

In classification the target is a category, here 1 for toxic and 0 for non-toxic. Classifiers usually output a **probability** for each class, and a threshold (by default 0.5) turns it into a decision. **Logistic regression**, despite its name, is the basic linear classifier: it passes a weighted sum of the features through an S-shaped function that maps any number into a probability between 0 and 1.

Errors now come in two kinds: a toxic molecule called safe (**false negative**) and a safe one called toxic (**false positive**). **Precision** is the fraction of predicted toxics that really are toxic, and **recall** is the fraction of real toxics that the model catches. A **confusion matrix** counts all four outcomes. The **ROC curve** shows the trade-off between catching toxics and raising false alarms as the threshold moves, and the area under it (ROC-AUC) summarises it in one number: 0.5 is random guessing, 1 is perfect.

`stratify=yc` keeps the same fraction of toxic molecules in the training and test sets, and `class_weight="balanced"` tells the models to pay more attention to the rare class.

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

Training can take minutes or hours, so a finished model is saved to disk and loaded when needed, instead of being retrained every time. `joblib` writes any scikit-learn model, including a whole pipeline with its scaler, to a single file. The reloaded model expects features in exactly the same order as during training.

```{code-cell} ipython3
import joblib
joblib.dump(grid.best_estimator_, "solubility_svr.joblib")
reloaded = joblib.load("solubility_svr.joblib")
print(reloaded.predict([[180.2, 1.3, 63.6, 1, 3]]))     # aspirin-like descriptors
```

```{admonition} Summary
:class: tip
- Supervised ML learns $X \rightarrow y$ from labelled examples: regression for numbers, classification for categories. Every model uses `fit` and `predict`.
- Split off a test set first and look at it once; tune hyperparameters with cross-validation on the training set.
- Regression metrics: MAE, RMSE, R²; check parity plots. Always compare with a simple baseline.
- Pipelines apply preprocessing such as scaling consistently and prevent leakage.
- Too simple underfits, too flexible overfits; no model can beat the noise in the data.
- For imbalanced classes, report precision, recall and ROC-AUC rather than accuracy.
```

## Exercises

**★ 1.** Retrain the linear model using only `logP`. How much worse is it?

**★★ 2.** Plot a learning curve (`sklearn.model_selection.learning_curve`) for the random forest. Would more data help?

**★★★ 3.** Predict `pIC50_sim` with the same descriptors. Which model is best, and why do all of them struggle? (Hint: look at how the value was simulated.)
