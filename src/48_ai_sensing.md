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

A single sensor is rarely selective. An **array** of partly selective sensors, read together by a pattern-recognition model, can be: this is the electronic nose, an idea that dates back to Persaud and Dodd {cite:p}`persaud1982`. The hard part in practice is not classification but **drift**: sensors age, and a model trained in January fails in June {cite:p}`vergara2012`. The principle is the same one a nose uses. No single receptor in the nose recognises coffee; a few hundred partly overlapping receptors each respond a little to many molecules, and the brain identifies the smell from the *pattern* across all of them. An **electronic nose** copies this with a handful of sensors, typically metal-oxide films whose electrical resistance changes when gases adsorb on them, and a model in place of the brain. The chemistry is poor, and deliberately so: a sensor that responded to one gas alone would be useless for everything else. The two things that make or break such a device are whether the array captures enough information to distinguish the analytes at all, and whether the model still works months after it was trained. This chapter simulates an eight-sensor array, shows how drift destroys a classifier, compares four countermeasures, quantifies concentrations and mixtures, and selects a smaller array.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
rng = np.random.default_rng(3)
```

## Simulating a sensor array

Eight metal-oxide sensors, each with its own affinity pattern, respond to six analytes. Responses follow a Langmuir isotherm in concentration, so the array is non-linear. A sensor's response to one gas is modelled as $a\,Kc/(1 + Kc)$, where $c$ is the concentration, $K$ the binding constant and $a$ the sensor's affinity for that gas (Chapter 26): the response rises almost linearly at low concentration and saturates once the surface is covered. The response to a mixture is the sum over gases. Saturation is the reason concentration cannot be read off a single number:

```{code-cell} ipython3
for c in [0.1, 1.0, 10.0]:
    print(c, round(2.0 * c / (1 + 2.0 * c), 3))
```

Two kinds of ageing are built in, because both happen to real sensors: the sensitivity decays exponentially, so every response shrinks over time, and the baseline wanders upward at a different rate for each sensor. There is also a small temperature cross-sensitivity and random noise.

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

Measurements are taken every other day for half a year, three repeats of each analyte at a random concentration, with a one-in-four chance that a second gas is present as an **interferent** and a room temperature that varies by a few degrees. That mixture of nuisances is what a deployed device faces, and keeping the day of each measurement makes the drift experiment below possible.

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

The crucial experiment: train on early data and test on **later** data, not on a random split. A random split puts measurements from the same week on both sides, so the model is tested on an array in the condition it was trained on, and the accuracy it reports is the accuracy on day one. Training on the first month and testing on the following five months is the question that matters, and the gap between the two numbers is a measure of how badly the device will age. This is the same reasoning as a scaffold split for molecules (Chapter 33): split along the axis the model will actually have to generalise across.

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

Each countermeasure attacks a different part of the problem. **Pattern normalisation** divides each sample's readings by their total, keeping only the relative shape of the response, which removes any drift that scales all sensors together. **Log-ratio features** take the logarithm of each sensor's reading divided by a reference sensor's, which cancels multiplicative drift exactly, because a common factor becomes a constant that the logarithm turns into a term subtracted from both:

```{code-cell} ipython3
raw = np.array([2.0, 4.0, 8.0])
print(np.log(raw / raw[0])[1:], np.log(0.5 * raw / (0.5 * raw[0]))[1:])   # halved gain, same features
```

**Recalibration** takes the opposite approach and accepts that the array changes, retraining each month on a couple of days of fresh measurements added to the original training set. The fourth option, keeping the raw features, is the baseline.

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

All three countermeasures beat the raw features, and monthly recalibration wins, but notice that the relative pattern does better than the log ratios. That is informative: the simulation drifts in two ways at once, a multiplicative loss of sensitivity and an additive baseline wander, and ratios cancel only the multiplicative part. Nothing here reaches the 92 % of the random split, which is the honest message of the section: drift is managed, not solved.

```{admonition} ⚗️ Chemistry note
:class: chem-note
The fix that works is almost always to make the **features** invariant to what drifts, rather than to make the model cleverer. If gain drifts, use ratios. If the baseline drifts, use differences from a reference gas. Then recalibrate on a schedule, and monitor accuracy as a quantity in its own right.
```

## Quantification, not just identification

Naming the gas is often not enough; a safety monitor has to say how much of it there is. That is a regression problem on the same array readings, and PLS (Chapter 39) is the standard tool, because the sensor responses are strongly correlated. The drift-robust log-ratio features are used here as well, and the model is fitted separately for each analyte.

```{code-cell} ipython3
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error, r2_score

fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, analyte, c in zip(axes, ["ethanol", "ammonia", "methane"], ["#6c4ad6", "#0f9d8a", "#e5487a"]):
    sub = sens[sens.analyte == analyte]
    Xs, ys = log_ratio(sub[SENSORS].values), sub.conc.values
    p = cross_val_predict(PLSRegression(4), Xs, ys, cv=KFold(5, shuffle=True, random_state=0)).ravel()
    print(f"{analyte:10s} R² = {r2_score(ys, p):.2f}, MAE = {mean_absolute_error(ys, p):.2f}")
    ax.scatter(ys, p, s=12, alpha=0.6, color=c)
    ax.plot([0, 4], [0, 4], "k--")
    ax.set(xlabel="true concentration", ylabel="predicted", aspect="equal",
           title=f"{analyte}: R² = {r2_score(ys, p):.2f}, MAE = {mean_absolute_error(ys, p):.2f}")
plt.tight_layout(); plt.show()
```

Ethanol is quantified reasonably (R² about 0.7) and methane badly (about 0.2), because only one sensor in the array responds to methane appreciably, and that one also responds to everything else. No model can recover information the hardware never captured, so sensor **selection** is part of model design.

## Mixtures: resolving two gases at once

A real sample is rarely one gas. Predicting the concentration of every analyte at once is a **multi-output** regression, which PLS handles natively by passing a matrix of targets instead of a vector. Whether it succeeds depends on whether each analyte leaves a distinguishable mark on the array: two gases that affect the same sensors in the same proportions cannot be separated no matter what model is used, a problem of the hardware rather than the mathematics.

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
print(scores.sort_values().round(2).to_string())
scores.sort_values().plot.barh(color="#3b82f6", figsize=(6.5, 3)); plt.xlabel("R² in mixtures")
plt.title("Mixture resolution is similar across the six analytes"); plt.show()
```

In this array every analyte comes out at roughly R² 0.6 to 0.7 in mixtures, with no analyte lost entirely, because each one has at least one sensor that prefers it. Methane does better here than in the single-analyte section only because the mixtures span a wider range of concentrations. An array with two near-identical sensors and nothing for a given gas would show the opposite, and testing for that is the point of the exercise.

## Designing a better array

Which sensors are worth their cost? Each sensor adds price, power and another thing to fail, so a device should carry the smallest array that does the job. **Greedy forward selection** answers directly: try every single sensor and keep the best, then try adding each of the rest and keep the best pair, and so on. It is not guaranteed to find the best subset of a given size, since a pair that works well together may contain no strong individual sensor, but it is cheap and usually close. The accuracy is measured on later data throughout, so the choice favours sensors that also drift gracefully.

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

The same mathematics applies to electrochemical sensors in a wearable patch or a point-of-care strip, where the extra constraints are power, calibration-free operation and per-person variability. An **amperometric** sensor measures the current produced when the analyte reacts at an electrode, which is proportional to its concentration over a useful range. The difficulty in a wearable is that the proportionality constant differs from person to person and from device to device, through skin properties, enzyme loading and electrode area, so the raw current is not comparable between users. A **one-point calibration**, a single reading against a reference measurement, estimates that person's gain and offset and brings the curves together, which is why glucose meters ask for a fingerstick reading.

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

```{admonition} Summary
:class: tip
- An array of non-selective sensors identifies analytes by the pattern of responses, not by any one reading.
- Langmuir saturation makes responses non-linear, so a single value cannot give a concentration.
- Validate against time: train early, test later. A random split hides drift completely.
- Make features invariant to what drifts (normalised patterns, log ratios), then recalibrate on a schedule.
- PLS quantifies single analytes and mixtures; what the hardware never captured, no model can recover.
- Greedy forward selection finds the smallest array that does the job.
```

## Reading further

- Persaud and Dodd's original model nose {cite:p}`persaud1982` introduced the array principle.
- Vergara and co-workers {cite:p}`vergara2012` quantified drift over 36 months and showed that classifier ensembles trained at different times mitigate it; their study is the standard reference for the experiment we repeated above.
- Chemometric preprocessing for drift follows the same logic as spectral preprocessing in Chapter 39 {cite:p}`wold2001`.

## Exercises

**★ 1.** Increase `DRIFT_RATE` tenfold. Which of the four countermeasures still works?

**★★ 2.** Add a humidity cross-sensitivity to `respond` and show that including measured humidity as an extra feature restores accuracy.

**★★★ 3.** Implement a simple domain-adaptation method: match the mean and covariance of the later data to the training data before classifying, and compare it with monthly recalibration.
