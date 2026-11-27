---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# AI-Assisted Discovery of Molecules and Materials

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Parts V and VI.
```

The previous parts built the pieces: models, representations, optimisers, generators. This chapter assembles them into the **discovery loop** that now runs in laboratories around the world: predict, select, make, measure, learn, repeat. Autonomous versions of this loop have discovered photocatalysts {cite:p}`burger2020` and synthesised new inorganic materials {cite:p}`szymanski2023`, while large-scale screening has proposed hundreds of thousands of candidate crystals {cite:p}`merchant2023`. The field's shape is well surveyed by Butler and co-workers {cite:p}`butler2018`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw, Descriptors, Crippen, QED, rdFingerprintGenerator, DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
from scipy.stats import norm
RDLogger.DisableLog("rdApp.*")
rng = np.random.default_rng(11)

DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name, **kw):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name, **kw)
df = load_data("molecules_sim.csv")
```

## The discovery loop

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7.5, 5))
steps = [("DESIGN\ngenerate or enumerate", "#6c4ad6"), ("PREDICT\nmodels + uncertainty", "#3b82f6"),
         ("SELECT\nacquisition function", "#0f9d8a"), ("MAKE\nsynthesis", "#f2a93b"),
         ("MEASURE\ncharacterisation", "#e5487a")]
angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, len(steps) + 1)[:-1]
for (label, colour), a in zip(steps, angles):
    x, y = np.cos(a), np.sin(a)
    ax.add_patch(plt.Circle((x, y), 0.33, color=colour, alpha=0.85))
    ax.text(x, y, label, ha="center", va="center", color="white", fontsize=9, weight="bold")
for a, b in zip(angles, np.roll(angles, -1)):
    ax.annotate("", xy=(np.cos(b) * 0.72, np.sin(b) * 0.72), xytext=(np.cos(a) * 0.72, np.sin(a) * 0.72),
                arrowprops=dict(arrowstyle="->", color="grey", lw=2, connectionstyle="arc3,rad=0.25"))
ax.text(0, 0, "LEARN", ha="center", va="center", fontsize=13, weight="bold")
ax.set(xlim=(-1.6, 1.6), ylim=(-1.6, 1.6)); ax.axis("off"); ax.set_aspect("equal")
ax.set_title("The closed-loop discovery cycle")
plt.show()
```

## A complete campaign, end to end

We now run a full closed-loop campaign on our simulated chemistry. The "laboratory" is a function that returns a measurement with noise; the agent may call it only a limited number of times. This is exactly the structure of a real project, with the advantage that we know the right answer and can score the search.

```{code-cell} ipython3
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
mols = [Chem.MolFromSmiles(s) for s in df.smiles]
X = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols], dtype=float)
fps = [gen.GetFingerprint(m) for m in mols]

# The hidden objective: potent, soluble, non-toxic and drug-like
true_objective = ((df.pIC50_sim - 5.0) / 2.0                      # potency
                  + 0.5 * np.clip(df.logS_sim + 4, 0, 2)          # solubility, saturating
                  - 1.5 * df.toxic_sim                            # safety penalty
                  + np.array([QED.qed(m) for m in mols])).values  # drug-likeness

def laboratory(i, noise=0.12):
    """One 'experiment': returns a noisy measurement of the objective for library member i."""
    return float(true_objective[i] + rng.normal(0, noise))

print(f"library size {len(df)},  best achievable objective {true_objective.max():.2f}")
```

```{code-cell} ipython3
def campaign(strategy, budget=120, n_init=12, seed=0, batch=6):
    r = np.random.default_rng(seed)
    measured = list(r.choice(len(df), n_init, replace=False))
    values = [laboratory(i) for i in measured]
    trace = [max(values)]
    while len(measured) < budget:
        unmeasured = np.setdiff1d(np.arange(len(df)), measured)
        if strategy == "random":
            pick = r.choice(unmeasured, batch, replace=False)
        else:
            # An ensemble of random forests gives prediction and uncertainty
            preds = []
            for s in range(5):
                rf = RandomForestRegressor(120, max_features=0.3, random_state=s, n_jobs=-1,
                                           bootstrap=True).fit(X[measured], values)
                preds.append(rf.predict(X[unmeasured]))
            preds = np.array(preds)
            mu, sd = preds.mean(0), preds.std(0) + 1e-6
            best = max(values)
            if strategy == "greedy":
                score = mu
            elif strategy == "explore":
                score = sd
            else:                                   # expected improvement
                z = (mu - best) / sd
                score = (mu - best) * norm.cdf(z) + sd * norm.pdf(z)
            pick = unmeasured[np.argsort(-score)[:batch]]
        for i in pick:
            measured.append(int(i)); values.append(laboratory(int(i)))
        trace.append(max(values))
    return np.array(trace), measured, values

strategies = {"random": "#9ca3af", "greedy": "#e5487a", "explore": "#3b82f6", "EI": "#0f9d8a"}
curves = {}
for s in strategies:
    runs = [campaign(s, seed=k)[0] for k in range(4)]
    curves[s] = np.array(runs)

fig, ax = plt.subplots(figsize=(8, 5))
x = 12 + np.arange(curves["random"].shape[1]) * 6
for s, c in strategies.items():
    m = curves[s].mean(0)
    ax.plot(x, m, color=c, lw=2.5, label=s)
    ax.fill_between(x, curves[s].min(0), curves[s].max(0), color=c, alpha=0.15)
ax.axhline(true_objective.max(), ls=":", color="k", label="best in library")
ax.set(xlabel="molecules measured", ylabel="best objective found", title="Four selection strategies, same budget")
ax.legend(); plt.show()
```

Pure exploitation (greedy) stalls on a local family; pure exploration wastes measurements; expected improvement, which balances the two, reaches the top of the library with a fraction of the experiments.

```{code-cell} ipython3
trace, measured, values = campaign("EI", budget=120, seed=1)
found = df.iloc[measured].assign(measured=values).sort_values("measured", ascending=False)
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in found.smiles[:8]], molsPerRow=4, subImgSize=(200, 160),
                     legends=[f"obj {v:.2f} | pIC50 {p:.1f}" for v, p in zip(found.measured[:8], found.pIC50_sim[:8])])
```

## Multi-objective optimisation and the Pareto front

Real projects never optimise one number. A molecule that is potent but insoluble is useless, and so is the reverse. The honest answer is not a single winner but the **Pareto front**: the set of molecules that no other molecule beats on every objective at once.

```{code-cell} ipython3
def pareto_front(points):
    """Indices of non-dominated points (maximising every column)."""
    keep = np.ones(len(points), dtype=bool)
    for i, p in enumerate(points):
        if keep[i]:
            dominated = np.all(points >= p, axis=1) & np.any(points > p, axis=1)
            keep[dominated] = False
    return np.where(keep)[0]

obj = np.column_stack([df.pIC50_sim, df.logS_sim, -df.toxic_sim.astype(float)])
front = pareto_front(obj)
print(f"{len(front)} of {len(df)} molecules are Pareto-optimal on potency, solubility and safety")

fig, ax = plt.subplots(figsize=(7.5, 5.5))
ax.scatter(df.pIC50_sim, df.logS_sim, s=8, c=df.toxic_sim, cmap="coolwarm", alpha=0.5)
f2 = pareto_front(np.column_stack([df.pIC50_sim, df.logS_sim]))
order = np.argsort(df.pIC50_sim.values[f2])
ax.plot(df.pIC50_sim.values[f2][order], df.logS_sim.values[f2][order], "-o", color="#0f9d8a", lw=2, ms=7,
        label="Pareto front (2 objectives)")
ax.set(xlabel="pIC50 (simulated)", ylabel="logS (simulated)", title="You must choose a point on the front")
ax.legend(); plt.show()
```

```{code-cell} ipython3
best_front = df.iloc[f2].sort_values("pIC50_sim", ascending=False).head(6)
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in best_front.smiles], molsPerRow=6, subImgSize=(170, 140),
                     legends=[f"{p:.1f} / {s:.1f}" for p, s in zip(best_front.pIC50_sim, best_front.logS_sim)])
```

### Scalarisation and hypervolume

Two standard ways to turn several objectives into a decision: combine them with weights (and vary the weights), or measure the **hypervolume** a candidate set dominates.

```{code-cell} ipython3
def hypervolume_2d(points, ref):
    """Area dominated by a set of 2D maximisation points relative to a reference corner."""
    pts = points[pareto_front(points)]
    pts = pts[np.argsort(-pts[:, 0])]
    hv, prev_y = 0.0, ref[1]
    for x, y in pts:
        if y > prev_y:
            hv += (x - ref[0]) * (y - prev_y)
            prev_y = y
    return hv

ref = np.array([df.pIC50_sim.min(), df.logS_sim.min()])
two = np.column_stack([df.pIC50_sim, df.logS_sim])
print(f"hypervolume of the whole library  = {hypervolume_2d(two, ref):.2f}")
print(f"hypervolume of the EI campaign    = {hypervolume_2d(two[measured], ref):.2f}")
print(f"hypervolume of a random 120       = {hypervolume_2d(two[rng.choice(len(df), 120, replace=False)], ref):.2f}")

fig, ax = plt.subplots(figsize=(7, 4.5))
weights = np.linspace(0, 1, 11)
for w in weights:
    score = w * (df.pIC50_sim - df.pIC50_sim.mean()) / df.pIC50_sim.std() + (1 - w) * (df.logS_sim - df.logS_sim.mean()) / df.logS_sim.std()
    i = int(np.argmax(score))
    ax.scatter(df.pIC50_sim[i], df.logS_sim[i], s=90, color=plt.cm.viridis(w), edgecolor="k", zorder=5)
ax.scatter(df.pIC50_sim, df.logS_sim, s=5, color="lightgrey")
ax.set(xlabel="pIC50", ylabel="logS", title="Each weighting picks a different winner (dark = potency-weighted)")
plt.show()
```

## Making the winners: synthesisability filters

A candidate that cannot be made is not a candidate. Cheap filters catch most of the nonsense before a chemist sees it.

```{code-cell} ipython3
ALERTS = {"reactive acyl halide": "C(=O)[Cl,Br,I]", "azide": "[N-]=[N+]=N", "peroxide": "[OX2][OX2]",
          "aldehyde": "[CX3H1]=O", "nitro": "[N+](=O)[O-]", "strained ring": "[r3]"}

def synthesis_flags(m):
    flags = [k for k, v in ALERTS.items() if m.HasSubstructMatch(Chem.MolFromSmarts(v))]
    return {"n_rings": Chem.rdMolDescriptors.CalcNumRings(m),
            "stereocentres": len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)),
            "heavy": m.GetNumHeavyAtoms(), "alerts": ";".join(flags), "n_alerts": len(flags)}

cand = found.head(30).copy()
flags = pd.DataFrame([synthesis_flags(Chem.MolFromSmiles(s)) for s in cand.smiles], index=cand.index)
cand = pd.concat([cand[["smiles", "measured"]], flags], axis=1)
passed = cand.query("n_alerts == 0 and stereocentres <= 2 and heavy <= 30")
print(f"{len(passed)} of the top {len(cand)} candidates pass the synthesisability filters")
cand.head(10)
```

## Learning from failure

The experiments that fail carry information too. A model trained on both successes and failures learns the boundary of the accessible space, which is what makes the next round smarter.

```{code-cell} ipython3
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# "Did the synthesis work?" simulated from size, stereocentres and alerts
heavy = np.array([m.GetNumHeavyAtoms() for m in mols])
stereo = np.array([len(Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)) for m in mols])
alerts = np.array([sum(m.HasSubstructMatch(Chem.MolFromSmarts(v)) for v in ALERTS.values()) for m in mols])
p_success = 1 / (1 + np.exp(-(3.0 - 0.07 * heavy - 0.8 * stereo - 0.9 * alerts)))
success = (rng.random(len(df)) < p_success).astype(int)
print(f"overall success rate {success.mean():.0%}")

Xs = np.column_stack([heavy, stereo, alerts, df.MW, df.RotB])
auc = cross_val_score(LogisticRegression(max_iter=1000), Xs, success, cv=5, scoring="roc_auc").mean()
print(f"synthesis-success model ROC-AUC = {auc:.2f}")

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].scatter(heavy, success + rng.normal(0, 0.03, len(df)), s=5, alpha=0.3, color="#6c4ad6")
bins = pd.cut(heavy, 8)
rate = pd.Series(success).groupby(bins, observed=True).mean()
axes[0].plot([i.mid for i in rate.index], rate.values, "-o", color="#e5487a", lw=2.5)
axes[0].set(xlabel="heavy atoms", ylabel="synthesis success", title="Bigger molecules fail more often")
# Expected value of an experiment = P(success) × predicted objective
rf = RandomForestRegressor(200, n_jobs=-1, random_state=0).fit(X[measured], values)
pred_obj = rf.predict(X)
p_ok = LogisticRegression(max_iter=1000).fit(Xs, success).predict_proba(Xs)[:, 1]
ev = p_ok * (pred_obj - pred_obj.min())
axes[1].scatter(pred_obj, p_ok, c=ev, cmap="viridis", s=8)
top_ev = np.argsort(-ev)[:10]
axes[1].scatter(pred_obj[top_ev], p_ok[top_ev], s=90, facecolor="none", edgecolor="k")
axes[1].set(xlabel="predicted objective", ylabel="P(synthesis succeeds)",
            title="Best expected value, not best prediction")
plt.tight_layout(); plt.show()
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
This is the single most useful idea in this chapter. A model that ranks by predicted property alone will send chemists after beautiful molecules that nobody can make. Ranking by *expected value*, the predicted gain times the probability of actually getting the measurement, is what makes a loop productive.
```

## What autonomous laboratories add

The published closed-loop systems differ in what they automate, not in the logic:

| System type | Design | Synthesis | Measurement | Human role |
|---|---|---|---|---|
| Virtual screening | computed | manual | manual | chooses everything to make |
| ML-guided project | model-proposed | manual | manual | approves each round |
| High-throughput + BO | model-proposed | robotic plate | automated reader | sets objective and bounds |
| Mobile robotic chemist {cite:p}`burger2020` | model-proposed | robot in a real lab | automated | sets objective, checks safety |
| Autonomous materials lab {cite:p}`szymanski2023` | computational screening | robotic furnaces | automated XRD | validates products |

```{code-cell} ipython3
scenarios = {"manual, 1 experiment / day": (1, 0.0), "manual + ML ranking": (1, 0.0),
             "HTE plate, 96 / week": (96 / 7, 0.0), "autonomous, 60 / day": (60, 0.0)}
budget_days = 60
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for (label, (rate, _)), c in zip(scenarios.items(), ["#9ca3af", "#3b82f6", "#f2a93b", "#0f9d8a"]):
    n = int(rate * budget_days)
    strategy = "random" if label.startswith("manual, ") else "EI"
    n = min(n, 400)
    tr, *_ = campaign(strategy, budget=max(n, 18), seed=3)
    days = np.linspace(0, min(budget_days, max(n, 18) / rate), len(tr))
    ax.plot(days, tr, color=c, lw=2.5, label=f"{label} ({strategy})")
ax.axhline(true_objective.max(), ls=":", color="k")
ax.set(xlabel="days of project time", ylabel="best objective found", xlim=(0, budget_days),
       title="Throughput and intelligence both help, and they multiply")
ax.legend(fontsize=8); plt.show()
```

## Case studies worth reading

All of these are open-literature reports; the citations point to the papers rather than reproducing their figures or data.

- **A mobile robotic chemist** {cite:p}`burger2020` searched a ten-variable photocatalyst space over eight days, running 688 experiments and finding mixtures six times more active than the starting point.
- **An autonomous inorganic laboratory** {cite:p}`szymanski2023` combined computational screening with robotic synthesis and automated X-ray diffraction to make dozens of previously unreported compounds.
- **Scaling deep learning for materials discovery** {cite:p}`merchant2023` used graph networks and active learning to predict large numbers of stable crystals, a scale impossible by DFT alone.
- **Bayesian reaction optimisation** {cite:p}`shields2021` showed that algorithmic optimisation matched or beat expert chemists on real reaction screens.
- **Deep learning for antibiotic discovery** {cite:p}`stokes2020` screened a large chemical library with a trained model and validated a structurally novel antibacterial compound experimentally.

```{admonition} How to read a discovery paper critically
:class: tip
Ask: what was the **baseline** (random selection? an expert?); how many experiments did the loop actually run; was the final result **validated independently** of the model; what was the **search space**, and would the answer have been found by enumeration; and are the data and code available so that the loop can be rerun?
```

## Exercises

**★ 1.** Rerun the campaign with a batch size of 1 and of 24 at the same total budget. Why does batch size matter?

**★★ 2.** Add a third objective (synthesis success probability) and compute the three-objective Pareto front. How many molecules survive?

**★★★ 3.** Implement a cost-aware acquisition function: each experiment has a cost drawn from the number of heavy atoms, and the budget is money rather than experiments. Does the best strategy change?
