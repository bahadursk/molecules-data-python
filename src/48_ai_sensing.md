---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# AI for Chemical Sensing: Electronic Noses and Tongues

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 31, 32 and 39.
```

A single sensor is rarely selective. An **array** of partly selective sensors, read together by a pattern-recognition model, can be: this is the electronic nose, an idea that dates back to Persaud and Dodd {cite:p}`persaud1982`. The hard part in practice is not classification but **drift**: sensors age, and a model trained in January fails in June {cite:p}`vergara2012`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
rng = np.random.default_rng(3)
```

## Simulating a sensor array

Eight metal-oxide sensors, each with its own affinity pattern, respond to six analytes. Responses follow a Langmuir isotherm in concentration, so the array is non-linear.

```{code-cell} ipython3
ANALYTES = ["ethanol", "acetone", "ammonia", "toluene", "H2S", "methane"]
SENSORS = [f"S{i+1}" for i in range(8)]
# Affinity matrix: how strongly each sensor binds each analyte (arbitrary units)
AFFINITY = np.array([
    [1.0, 0.7, 0.2, 0.5, 0.1, 0.05],
    [0.6, 1.0, 0.1, 0.8, 0.2, 0.10],
    [0.1, 0.2, 1.0, 0.1, 0.9, 0.05],
    [0.4, 0.3, 0.8, 0.2, 1.0, 0.10],
    [0.8, 0.5, 0.3, 1.0, 0.3, 0.30],
    [0.2, 0.1, 0.2, 0.3, 0.2, 1.00],
    [0.9, 0.9, 0.5, 0.7, 0.6, 0.20],
    [0.3, 0.6, 0.6, 0.4, 0.8, 0.50]])
K = rng.uniform(0.5, 3.0, (8, 6))                  # Langmuir constants
GAIN = rng.uniform(0.8, 1.2, 8)

AGEING_TAU = 400.0                                  # days: sensitivity decay time constant
DRIFT_RATE = np.linspace(0.002, 0.02, 8)            # baseline drift per day, different per sensor

def respond(concentrations, t_days=0.0, temperature=25.0, noise=0.02):
    """Array response to a mixture, with ageing drift and a temperature cross-sensitivity."""
    c = np.asarray(concentrations, dtype=float)
    r = (AFFINITY * (K * c) / (1 + K * c)).sum(axis=1)
    sensitivity_loss = np.exp(-t_days / AGEING_TAU)        # sensors lose sensitivity as they age
    baseline_drift = DRIFT_RATE * t_days                   # and their baselines wander
    r = GAIN * sensitivity_loss * r + baseline_drift
    r *= 1 + 0.004 * (temperature - 25.0)
    return r * (1 + rng.normal(0, noise, 8))

fig, ax = plt.subplots(figsize=(8, 4))
patterns = np.array([respond(np.eye(6)[i] * 2.0, noise=0.005) for i in range(6)])
sns.heatmap(pd.DataFrame(patterns, index=ANALYTES, columns=SENSORS), cmap="magma", annot=True, fmt=".2f",
            cbar_kws={"label": "response"}, ax=ax)
ax.set_title("Each analyte gives a pattern, not a single reading"); plt.tight_layout(); plt.show()
```

## Building a dataset

```{code-cell} ipython3
rows = []
for day in range(0, 180, 2):
    for label, name in enumerate(ANALYTES):
        for _ in range(3):
            conc = np.zeros(6)
            conc[label] = rng.uniform(0.3, 4.0)
            if rng.random() < 0.25:                                   # sometimes an interferent is present
                other = rng.integers(6)
                conc[other] += rng.uniform(0.1, 0.8)
            T = rng.normal(25, 4)
            rows.append({"analyte": name, "day": day, "T": T, "conc": conc[label],
                         **dict(zip(SENSORS, respond(conc, t_days=day, temperature=T)))})
sens = pd.DataFrame(rows)
print(sens.shape)
sens.head(3).round(3)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for s, c in zip(SENSORS, plt.cm.viridis(np.linspace(0, 0.9, 8))):
    axes[0].plot(sens.groupby("day")[s].mean(), color=c, label=s)
axes[0].set(xlabel="day", ylabel="mean response", title="Drift: the array changes under your feet")
axes[0].legend(ncol=2, fontsize=7)
sub = sens[sens.analyte.isin(["ethanol", "ammonia"])]
sns.scatterplot(sub, x="S1", y="S3", hue="analyte", size="day", sizes=(8, 60), alpha=0.7, ax=axes[1])
axes[1].set_title("Two-sensor view: classes separate, then move")
plt.tight_layout(); plt.show()
```

## Classification, and how drift destroys it

The crucial experiment: train on early data and test on **later** data, not on a random split.

```{code-cell} ipython3
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay

X = sens[SENSORS].values
y = sens.analyte.values
early = sens.day < 30

def fit_eval(Xtr, ytr, Xte, yte):
    clf = make_pipeline(StandardScaler(), RandomForestClassifier(300, n_jobs=-1, random_state=0)).fit(Xtr, ytr)
    return clf, accuracy_score(yte, clf.predict(Xte))

Xa, Xb, ya, yb = train_test_split(X, y, test_size=0.3, random_state=0)
_, acc_random = fit_eval(Xa, ya, Xb, yb)
clf_time, acc_time = fit_eval(X[early], y[early], X[~early], y[~early])
print(f"random split accuracy        {acc_random:.2%}")
print(f"train early, test later      {acc_time:.2%}")

by_month = [(d, accuracy_score(y[(sens.day >= d) & (sens.day < d + 30)],
                               clf_time.predict(X[(sens.day >= d) & (sens.day < d + 30)])))
            for d in range(0, 180, 30)]
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].plot(*zip(*by_month), "o-", color="#e5487a", lw=2.5)
axes[0].set(xlabel="day of test data", ylabel="accuracy", ylim=(0, 1), title="A model trained once decays")
ConfusionMatrixDisplay.from_predictions(y[sens.day > 150], clf_time.predict(X[sens.day > 150]), ax=axes[1],
                                        cmap="Purples", colorbar=False, xticks_rotation=60)
axes[1].set_title("Errors after six months")
plt.tight_layout(); plt.show()
```

## Four ways to fight drift

```{code-cell} ipython3
def normalise_within_sample(Xraw):
    """Relative pattern only: divide by the total response. Removes common-mode gain drift."""
    return Xraw / Xraw.sum(axis=1, keepdims=True)

def log_ratio(Xraw):
    """Log-ratio to the first sensor: removes multiplicative drift."""
    return np.log(np.clip(Xraw, 1e-6, None) / np.clip(Xraw[:, [0]], 1e-6, None))[:, 1:]

results = {}
# 1. raw
results["raw features"] = acc_time
# 2. pattern normalisation
Xn = normalise_within_sample(X)
_, results["normalised pattern"] = fit_eval(Xn[early], y[early], Xn[~early], y[~early])
# 3. log ratios
Xl = log_ratio(X)
_, results["log-ratio features"] = fit_eval(Xl[early], y[early], Xl[~early], y[~early])
# 4. periodic recalibration: retrain every 30 days on a few fresh samples
acc_recal = []
for d in range(30, 180, 30):
    recal = (sens.day >= d - 2) & (sens.day < d)
    mask_tr = early | recal
    test = (sens.day >= d) & (sens.day < d + 30)
    _, a = fit_eval(X[mask_tr], y[mask_tr], X[test], y[test])
    acc_recal.append(a)
results["recalibrated monthly"] = float(np.mean(acc_recal))

pd.Series(results).sort_values().plot.barh(color=["#9ca3af", "#f2a93b", "#3b82f6", "#0f9d8a"], figsize=(7, 3.2))
plt.xlabel("accuracy on later data"); plt.xlim(0, 1); plt.title("Drift-robust features and recalibration")
plt.show()
pd.Series(results).round(3)
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
The fix that works is almost always to make the **features** invariant to what drifts, rather than to make the model cleverer. If gain drifts, use ratios. If the baseline drifts, use differences from a reference gas. Then recalibrate on a schedule, and monitor accuracy as a quantity in its own right.
```

## Quantification, not just identification

```{code-cell} ipython3
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, analyte, c in zip(axes, ["ethanol", "ammonia", "methane"], ["#6c4ad6", "#0f9d8a", "#e5487a"]):
    sub = sens[sens.analyte == analyte]
    Xs, ys = log_ratio(sub[SENSORS].values), sub.conc.values
    p = cross_val_predict(PLSRegression(4), Xs, ys, cv=KFold(5, shuffle=True, random_state=0)).ravel()
    ax.scatter(ys, p, s=12, alpha=0.6, color=c)
    ax.plot([0, 4], [0, 4], "k--")
    ax.set(xlabel="true concentration", ylabel="predicted", aspect="equal",
           title=f"{analyte}: R² = {r2_score(ys, p):.2f}, MAE = {mean_absolute_error(ys, p):.2f}")
plt.tight_layout(); plt.show()
```

Methane is quantified poorly because only one sensor in the array responds to it appreciably: no model can recover information the hardware never captured. Sensor **selection** is part of model design.

## Mixtures: resolving two gases at once

```{code-cell} ipython3
mix_rows = []
for _ in range(900):
    c = np.zeros(6)
    i, j = rng.choice(6, 2, replace=False)
    c[i], c[j] = rng.uniform(0, 3), rng.uniform(0, 3)
    mix_rows.append({**dict(zip(SENSORS, respond(c, t_days=rng.uniform(0, 60)))), **{f"c_{a}": v for a, v in zip(ANALYTES, c)}})
mix = pd.DataFrame(mix_rows)
Xm = log_ratio(mix[SENSORS].values)
Ym = mix[[f"c_{a}" for a in ANALYTES]].values
pred = cross_val_predict(PLSRegression(6), Xm, Ym, cv=5)
scores = pd.Series({a: r2_score(Ym[:, k], pred[:, k]) for k, a in enumerate(ANALYTES)})
scores.sort_values().plot.barh(color="#3b82f6", figsize=(6.5, 3)); plt.xlabel("R² in mixtures")
plt.title("Some analytes stay resolvable in mixtures, others do not"); plt.show()
```

## Designing a better array

Which sensors are worth their cost? Greedy forward selection answers directly.

```{code-cell} ipython3
chosen, remaining = [], list(range(8))
history = []
while remaining:
    best, best_acc = None, -1
    for s in remaining:
        cols = chosen + [s]
        Xc = X[:, cols]
        _, a = fit_eval(Xc[early], y[early], Xc[~early], y[~early])
        if a > best_acc:
            best, best_acc = s, a
    chosen.append(best); remaining.remove(best)
    history.append((len(chosen), best_acc, SENSORS[best]))

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot([h[0] for h in history], [h[1] for h in history], "o-", color="#0f9d8a", lw=2.5)
for n, a, name in history:
    ax.annotate(name, (n, a), textcoords="offset points", xytext=(0, -14), ha="center", fontsize=8)
ax.set(xlabel="number of sensors in the array", ylabel="accuracy on later data",
       title="Three well-chosen sensors do most of the work")
plt.show()
```

## Wearable and point-of-care sensing

The same mathematics applies to electrochemical sensors in a wearable patch or a point-of-care strip, where the extra constraints are power, calibration-free operation and per-person variability.

```{code-cell} ipython3
# A glucose-like amperometric sensor: current vs concentration, with person-to-person gain variation
conc = np.linspace(0, 20, 60)                             # mmol/L
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for person in range(6):
    gain = rng.normal(1.0, 0.18)
    offset = rng.normal(0, 0.4)
    i = gain * 3.2 * conc / (1 + conc / 25) + offset + rng.normal(0, 0.15, conc.size)
    axes[0].plot(conc, i, color=plt.cm.tab10(person), alpha=0.8)
    axes[1].plot(conc, (i - offset) / gain, color=plt.cm.tab10(person), alpha=0.8)
axes[0].set(xlabel="glucose / mM", ylabel="current / μA", title="Six individuals, uncalibrated")
axes[1].set(xlabel="glucose / mM", ylabel="calibrated signal", title="After a one-point personal calibration")
plt.tight_layout(); plt.show()
```

```{warning}
A sensor that reports a clinical number is a medical device. Accuracy claims need prospective validation against a reference method, and a model that drifts or that was validated on one population can cause real harm. Never present a classroom model's output as a measurement.
```

## Reading further

- Persaud and Dodd's original model nose {cite:p}`persaud1982` introduced the array principle.
- Vergara and co-workers {cite:p}`vergara2012` quantified drift over 36 months and showed that classifier ensembles trained at different times mitigate it; their study is the standard reference for the experiment we repeated above.
- Chemometric preprocessing for drift follows the same logic as spectral preprocessing in Chapter 39 {cite:p}`wold2001`.

## Exercises

**★ 1.** Increase `DRIFT_RATE` tenfold. Which of the four countermeasures still works?

**★★ 2.** Add a humidity cross-sensitivity to `respond` and show that including measured humidity as an extra feature restores accuracy.

**★★★ 3.** Implement a simple domain-adaptation method: match the mean and covariance of the later data to the training data before classifying, and compare it with monthly recalibration.
