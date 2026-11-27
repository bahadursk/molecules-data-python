---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Trustworthy, Explainable and Open AI for Science

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 31, 34, 36 and 43. The final chapter of the book.
```

Fifty chapters have taught you to build models. This one is about deserving to be believed. A model that enters a real decision — which compound to make, which batch to release, which paper to write — acquires obligations: it must be able to say *why*, to say *how sure*, to say *when it is out of its depth*, and to be checked by someone who was not in the room when it was trained. None of that is optional extra credit. It is the difference between computational chemistry and computational storytelling.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
rng = np.random.default_rng(51)

DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"

def load_data(name, **kw):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name, **kw)

mol = load_data("molecules_sim.csv")
print(mol.shape)
mol.head(3)
```

## The four questions a trustworthy model must answer

| Question | Tool | Chapter |
|---|---|---|
| Why this prediction? | feature attribution, surrogate rules, mechanistic sanity checks | this one, 43 |
| How sure are you? | ensembles, Gaussian processes, calibration curves | 31, 35, this one |
| Is this input even in scope? | applicability domain, novelty detection | 36, this one |
| Can I reproduce you? | pinned environments, seeds, FAIR data and code | 2, this one |

A model that answers all four can still be wrong — but it will be wrong *visibly*, which is all science has ever asked of a method.

## A model to interrogate

The model we will interrogate is a **hit caller**: it flags the most potent quarter of the library (simulated pIC$_{50}$ above its upper quartile) from five physicochemical descriptors, with a scaffold-aware split so that the test molecules are structurally new (Chapter 36). Potency in our simulated library depends on a hinge-binding motif that these descriptors only partly capture, so the model is good but not perfect, which is exactly the situation in which trust matters. Everything that follows is diagnosis of this one model.

```{code-cell} ipython3
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score

FEATS = [c for c in ["MW", "logP", "TPSA", "HBD", "RotB"] if c in mol.columns]
print("features:", FEATS)

# group split on scaffold: fill the test set with whole scaffolds until it holds ~25% of rows
groups = mol["scaffold"].fillna("(acyclic)").astype(str)     # acyclic molecules have an empty scaffold
order = list(pd.unique(groups))
np.random.default_rng(1).shuffle(order)          # a separate, fixed RNG for the split
sizes = groups.value_counts()
test_groups, n = set(), 0
for g in order:
    if n >= 0.25 * len(mol):
        break
    test_groups.add(g); n += int(sizes[g])
is_test = groups.isin(test_groups).to_numpy()

X = mol[FEATS].astype(float)
P_CUT = float(mol["pIC50_sim"].quantile(0.75))
print(f"label: pIC50_sim > {P_CUT:.2f}")
y = (mol["pIC50_sim"] > P_CUT).astype(int)
Xtr, Xte = X[~is_test], X[is_test]
ytr, yte = y[~is_test], y[is_test]
print(f"train {len(ytr)}  test {len(yte)}  positives {ytr.mean():.2%} / {yte.mean():.2%}")
# group splits can silently unbalance the classes: always print this line

clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                             random_state=0, n_jobs=-1).fit(Xtr, ytr)
p_te = clf.predict_proba(Xte)[:, 1]
print(f"accuracy {accuracy_score(yte, p_te > 0.5):.3f}   ROC-AUC {roc_auc_score(yte, p_te):.3f}")
```

## Explainability

### Global: which features does the model use?

Impurity-based importances are biased towards high-cardinality features. **Permutation importance** measures what actually happens to held-out performance when you destroy a feature, which is the question you meant to ask.

```{code-cell} ipython3
from sklearn.inspection import permutation_importance

pi = permutation_importance(clf, Xte, yte, n_repeats=15, random_state=0,
                            scoring="roc_auc", n_jobs=-1)
imp = (pd.DataFrame({"feature": FEATS, "drop_in_AUC": pi.importances_mean,
                     "sd": pi.importances_std})
       .sort_values("drop_in_AUC", ascending=False))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
axes[0].barh(imp.feature[::-1], imp.drop_in_AUC[::-1], xerr=imp.sd[::-1],
             color="#0f9d8a")
axes[0].set(xlabel="loss of test ROC-AUC when shuffled", title="permutation importance")
axes[1].barh(np.array(FEATS)[np.argsort(clf.feature_importances_)],
             np.sort(clf.feature_importances_), color="#f2a93b")
axes[1].set(xlabel="mean impurity decrease", title="built-in importance (biased)")
plt.tight_layout(); plt.show()
imp.round(4)
```

### Local: why *this* molecule?

A global ranking does not help a medicinal chemist holding one compound. Shapley values distribute a single prediction's deviation from the average across the features, with useful additivity guarantees {cite:p}`lundberg2017`. We compute an exact Shapley decomposition for a tree ensemble the slow, transparent way — by brute force over feature subsets — so you can see there is no magic in it.

```{code-cell} ipython3
from itertools import combinations
from math import comb

def shapley_exact(model, x, background, feats, max_feats=8):
    """Exact Shapley values by marginalising absent features over a background sample."""
    x = np.asarray(x, float)
    bg = np.asarray(background, float)
    d = len(feats)
    assert d <= max_feats, "brute force is 2^d; keep d small"

    cache = {}
    def v(S):
        key = tuple(sorted(S))
        if key not in cache:
            Z = bg.copy()
            for j in key:
                Z[:, j] = x[j]
            cache[key] = model.predict_proba(pd.DataFrame(Z, columns=feats))[:, 1].mean()
        return cache[key]

    phi = np.zeros(d)
    allf = range(d)
    for j in allf:
        rest = [k for k in allf if k != j]
        for size in range(len(rest) + 1):
            w = 1.0 / (d * comb(d - 1, size))
            for S in combinations(rest, size):
                phi[j] += w * (v(list(S) + [j]) - v(list(S)))
    return phi, v([]), v(list(allf))

bg = Xtr.sample(120, random_state=0).to_numpy()
idx = int(np.argmax(p_te))                            # the molecule it is most confident about
x0 = Xte.iloc[idx]
phi, base, full = shapley_exact(clf, x0, bg, FEATS)

print(f"baseline P(hit) = {base:.3f}")
print(f"this molecule     = {full:.3f}")
print(f"sum of phi + base = {base + phi.sum():.3f}   (additivity check)")

order = np.argsort(np.abs(phi))
fig, ax = plt.subplots(figsize=(7.4, 4.2))
cols = ["#e5487a" if p > 0 else "#0f9d8a" for p in phi[order]]
ax.barh([f"{FEATS[i]} = {x0[FEATS[i]]:.1f}" for i in order], phi[order], color=cols)
ax.axvline(0, color="k", lw=1)
ax.set(xlabel="contribution to P(hit)", title="local explanation for one molecule")
plt.tight_layout(); plt.show()
```

```{admonition} ⚗️ Chemistry first
:class: chem-note
Read a local explanation as a hypothesis, not a mechanism. "High logP pushed this prediction up" is a statement about the *model*; whether lipophilicity causes the potency is a statement about *biology*, and the model has no access to that. The test of an explanation is whether a chemist can turn it into an experiment: make the analogue with lower logP, measure, and see if the prediction's story survives.
```

### Explanations can agree and still be wrong

A sobering exercise: train the same model on a deliberately leaky feature and watch the explanation confidently point at it.

```{code-cell} ipython3
leak = mol[FEATS].astype(float).copy()
leak["batch_id"] = (y.to_numpy() * 1000
                    + rng.integers(0, 60, len(mol)))      # an "ID" that encodes the label
Ltr, Lte = leak[~is_test], leak[is_test]
leaky = RandomForestClassifier(n_estimators=400, random_state=0, n_jobs=-1).fit(Ltr, ytr)
pl = leaky.predict_proba(Lte)[:, 1]
print(f"leaky model ROC-AUC {roc_auc_score(yte, pl):.3f}  vs honest model {roc_auc_score(yte, p_te):.3f}")

pil = permutation_importance(leaky, Lte, yte, n_repeats=8, random_state=0,
                             scoring="roc_auc", n_jobs=-1)
print(pd.Series(pil.importances_mean, index=leak.columns)
      .sort_values(ascending=False).round(3).to_string())
```

The AUC is near-perfect and the explanation is honest about *what the model used* — a batch identifier. Explainability did not prevent the disaster; it diagnosed it, and only because a human looked at the feature names and knew that `batch_id` is not chemistry. **Interpretability is a tool for sceptical humans, not a substitute for one.**

## Uncertainty you can act on

### Calibration

A probability of 0.8 should be right 80% of the time. Check it, do not assume it.

```{code-cell} ipython3
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.metrics import brier_score_loss

frac, mean_pred = calibration_curve(yte, p_te, n_bins=8, strategy="quantile")
cal = CalibratedClassifierCV(RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                                    random_state=0),
                             method="isotonic", cv=4).fit(Xtr, ytr)
p_cal = cal.predict_proba(Xte)[:, 1]
frac_c, mean_c = calibration_curve(yte, p_cal, n_bins=8, strategy="quantile")

fig, ax = plt.subplots(figsize=(5.8, 5.2))
ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
ax.plot(mean_pred, frac, "o-", color="#e5487a",
        label=f"raw RF (Brier {brier_score_loss(yte, p_te):.3f})")
ax.plot(mean_c, frac_c, "s-", color="#0f9d8a",
        label=f"isotonic (Brier {brier_score_loss(yte, p_cal):.3f})")
ax.set(xlabel="predicted probability", ylabel="observed frequency", title="calibration")
ax.legend(); plt.tight_layout(); plt.show()
```

```{code-cell} ipython3
print(f"ROC-AUC  raw {roc_auc_score(yte, p_te):.3f}   isotonic {roc_auc_score(yte, p_cal):.3f}")
print(f"Brier    raw {brier_score_loss(yte, p_te):.4f}  isotonic {brier_score_loss(yte, p_cal):.4f}")
```

Calibration is a monotone re-mapping, so it barely moves the AUC: the *ranking* of molecules stays the same. What it changes is whether the *numbers* mean what they say, and that shows up in the Brier score and in how closely the curve hugs the diagonal. Whether it helps depends on the model and the split (random forests are often under-confident at the extremes; on a structurally novel test set a calibrator fitted on training scaffolds may not transfer), so read your own numbers above rather than assuming. Calibrated numbers are what you need if you plan to say "we will synthesise everything above 0.3".

### Two kinds of uncertainty

*Aleatoric* uncertainty is noise in the system: more data will not remove it. *Epistemic* uncertainty is ignorance: more data will. Ensemble disagreement estimates the second, which is the one that tells you where to look next.

```{code-cell} ipython3
per_tree = np.stack([t.predict_proba(Xte.to_numpy())[:, 1] for t in clf.estimators_])
mean_p = per_tree.mean(0)
epistemic = per_tree.std(0)
aleatoric = np.sqrt(mean_p * (1 - mean_p))

fig, axes = plt.subplots(1, 2, figsize=(13, 4.3))
axes[0].scatter(mean_p, epistemic, s=12, alpha=.5, color="#6c4ad6", label="epistemic (tree spread)")
axes[0].scatter(mean_p, aleatoric, s=12, alpha=.4, color="#f2a93b", label="aleatoric (√p(1-p))")
axes[0].set(xlabel="mean predicted P(hit)", ylabel="uncertainty",
            title="the two uncertainties behave differently")
axes[0].legend()

# does high uncertainty actually mark the errors? selective prediction
err = (mean_p > 0.5).astype(int) != yte.to_numpy()
ordr = np.argsort(epistemic)
cov = np.arange(1, len(err) + 1) / len(err)
risk = np.cumsum(err[ordr]) / np.arange(1, len(err) + 1)
axes[1].plot(cov, risk, color="#0f9d8a", lw=2, label="abstain on most uncertain")
axes[1].axhline(err.mean(), ls="--", c="grey", label=f"predict everything ({err.mean():.3f})")
axes[1].set(xlabel="coverage (fraction answered)", ylabel="error rate on answered",
            title="risk–coverage curve")
axes[1].legend(); plt.tight_layout(); plt.show()

for c in [0.5, 0.7, 0.9]:
    k = int(c * len(err))
    print(f"answering the {c:.0%} most confident: error {risk[k-1]:.3f}")
```

A downward-sloping risk–coverage curve is the proof that the uncertainty estimate is worth something: if you let the model say "I don't know" on the hardest fifth of cases, the error on the rest falls. This is the single most practical use of uncertainty in a screening campaign.

## Out-of-distribution detection

Calibration and explanations assume the input is the sort of thing the model has seen. Checking that is a separate, cheap, and routinely skipped step.

```{code-cell} ipython3
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

sc = StandardScaler().fit(Xtr)
iso = IsolationForest(n_estimators=300, random_state=0).fit(sc.transform(Xtr))
nn = NearestNeighbors(n_neighbors=5).fit(sc.transform(Xtr))

def ood_scores(df):
    Z = sc.transform(df)
    d, _ = nn.kneighbors(Z)
    return -iso.score_samples(Z), d.mean(1)

# a deliberately out-of-domain set: inorganic-ish, very large, very polar oddities
COLUMNS_OOD = {"MW":   [1400, 95, 15, 60, 800],
               "logP": [-6, 11, 2, 0.5, 1],
               "TPSA": [420, 5, 10, 60, 700],
               "HBD":  [14, 0, 1, 2, 20],
               "RotB": [30, 0, 1, 3, 45]}
weird = pd.DataFrame({f: COLUMNS_OOD[f] for f in FEATS})
weird

iso_in, knn_in = ood_scores(Xte)
iso_out, knn_out = ood_scores(weird)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
for ax, (a, b, name) in zip(axes, [(iso_in, iso_out, "isolation-forest score"),
                                   (knn_in, knn_out, "mean distance to 5 NN")]):
    ax.hist(a, bins=40, color="#0f9d8a", alpha=.75, label="test molecules")
    for v in b:
        ax.axvline(v, color="#e5487a", lw=1.5)
    ax.set(xlabel=name, ylabel="count", title=f"OOD flagging: {name}")
    ax.legend()
plt.tight_layout(); plt.show()

print("model's confidence on the out-of-domain inputs:")
print(np.round(clf.predict_proba(weird)[:, 1], 3))
```

Look at that last line. The model returns confident-looking probabilities for molecules it has no business judging. It cannot refuse; it has no mechanism for refusal. **You** have to add one — a gate that compares every new input against the training distribution and declines to answer outside it. Two lines of code, routinely omitted, and the commonest reason a published model fails on someone else's compounds.

```{admonition} 🤖 ML corner
:class: ml-corner
A deployment-ready predictor is not a model. It is a *pipeline*: input validation → domain check → prediction → uncertainty → calibration → a refusal path. Wrap those steps in one object with one `predict` method and the discipline becomes automatic instead of aspirational.
```

```{code-cell} ipython3
class GatedPredictor:
    """A predictor that is allowed to say 'I don't know'."""

    def __init__(self, model, scaler, nn, feats, knn_cut, unc_cut):
        self.m, self.sc, self.nn, self.feats = model, scaler, nn, feats
        self.knn_cut, self.unc_cut = knn_cut, unc_cut

    def predict(self, df):
        missing = [f for f in self.feats if f not in df.columns]
        if missing:
            raise ValueError(f"missing features: {missing}")
        raw = df[self.feats].astype(float)
        d, _ = self.nn.kneighbors(self.sc.transform(raw))     # domain check in scaled space
        dist = d.mean(1)
        per = np.stack([t.predict_proba(raw.to_numpy())[:, 1]  # trees see raw features,
                        for t in self.m.estimators_])          # exactly as in training
        p, u = per.mean(0), per.std(0)
        out = pd.DataFrame({"p_hit": p, "uncertainty": u, "distance": dist})
        out["verdict"] = np.where(dist > self.knn_cut, "refused: outside domain",
                           np.where(u > self.unc_cut, "refused: too uncertain",
                             np.where(p > 0.5, "flagged", "clear")))
        return out

gp_ = GatedPredictor(clf, sc, nn, FEATS,
                     knn_cut=float(np.percentile(knn_in, 99)),
                     unc_cut=float(np.percentile(epistemic, 95)))
print(gp_.predict(weird).round(3).to_string(index=False))
print()
res = gp_.predict(Xte)
print(res.verdict.value_counts().to_string())
answered = res.verdict.isin(["flagged", "clear"])
print(f"\naccuracy on answered ({answered.mean():.0%} of cases): "
      f"{accuracy_score(yte[answered.to_numpy()], (res.p_hit[answered] > 0.5)):.3f}")
```

Four of the five odd inputs are refused, and the one that passes is the small, unremarkable polar molecule (MW 60, logP 0.5) that genuinely sits inside the training distribution: the gate is not refusing things for looking unusual on paper, only for being far from anything it has seen. On the real test set it declines a few percent of cases, and accuracy on the rest rises. That trade (answer fewer, be right more often) is the whole point.

## FAIR data and reproducible science

The FAIR principles — Findable, Accessible, Interoperable, Reusable — were written for data stewardship {cite:p}`wilkinson2016` and apply just as well to models. In practice, for a chemistry ML paper, they reduce to a short checklist you can actually complete.

| Requirement | What it means for your project |
|---|---|
| **F**indable | the dataset and code have a persistent identifier (a DOI from a repository such as Zenodo), not just a lab URL |
| **A**ccessible | downloadable without a login or an email to the corresponding author |
| **I**nteroperable | standard formats: SMILES or InChI {cite:p}`weininger1988,heller2015` for molecules, CIF for structures, CSV or JSON for tables, with units stated |
| **R**eusable | an explicit licence, a data dictionary, the train/test split you used, and a pinned environment |

```{code-cell} ipython3
import sys, platform, sklearn, numpy, pandas, matplotlib, json

env = {"python": sys.version.split()[0], "platform": platform.platform(),
       "numpy": numpy.__version__, "pandas": pandas.__version__,
       "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__}
print(json.dumps(env, indent=2))
```

```{code-cell} ipython3
# A model card: the minimum metadata a reader needs to judge, reuse or refuse your model.
card = {
    "name": "hit-calling classifier (teaching example)",
    "purpose": "demonstrate trustworthy-ML practice; NOT a potency predictor for real use",
    "training_data": f"molecules_sim.csv — label is simulated pIC50_sim > {P_CUT:.2f}",
    "n_train": int(len(ytr)), "n_test": int(len(yte)),
    "split": "scaffold group split (structurally disjoint test set)",
    "features": FEATS,
    "metrics": {"roc_auc": round(float(roc_auc_score(yte, p_te)), 3),
                "accuracy": round(float(accuracy_score(yte, p_te > 0.5)), 3),
                "brier": round(float(brier_score_loss(yte, p_te)), 3)},
    "applicability_domain": f"mean 5-NN distance <= {np.percentile(knn_in, 99):.2f} in standardised feature space",
    "known_limitations": ["labels are simulated, not measured",
                          "descriptor-only: no 3-D or target information",
                          "no external validation set"],
    "licence": "CC-BY-4.0 (text and data); MIT (code)",
    "environment": env,
}
print(json.dumps(card, indent=2)[:1200] + "\n...")
```

```{admonition} ⚗️ Chemistry first
:class: chem-note
Write the model card *before* you tune the model. It forces you to state the intended use and the split up front, which is precisely when it is still cheap to notice that your split leaks, your labels are unmeasured, or your intended use was never testable.
```

## Ethics, safety and dual use

Three concerns deserve naming explicitly, because chemistry ML touches all of them.

**Dual use.** A generative model trained to find potent molecules is, by construction, a model that can be asked for toxic ones; Urbina and colleagues documented exactly this inversion and the field's unpreparedness for it {cite:p}`urbina2022`. The responsible pattern is not to stop building such models but to treat capability and release as separate decisions: hazard screening in the generation loop, restraint about publishing ready-to-run weights for hazard-relevant targets, and institutional review for work near the line. If a project you are on heads that way, escalate it rather than deciding alone.

**Bias and silent exclusion.** Chemical datasets are not random samples of chemical space. They are records of what people chose to make, which reflects commercial priorities, available reagents and fashion. A model trained on them will rate familiar chemistry as good and unfamiliar chemistry as unpredictable, and a screening campaign that trusts it will narrow over time. The countermeasures are diversity-aware selection (Chapter 47), reporting performance per scaffold class rather than only in aggregate, and keeping a random-exploration quota in every campaign.

**Automation and accountability.** As models drive instruments {cite:p}`burger2020,boiko2023` and language models call tools {cite:p}`bran2024,jablonka2024`, the chain between a prediction and a physical action gets shorter. Someone must still be responsible for what comes out of the fume hood. Write down who, with what veto, before the loop closes — not after.

```{admonition} A standard worth holding yourself to
:class: warning
Before a model of yours influences a real decision, you should be able to answer, out loud, without looking anything up: what it was trained on, how the test set was separated, what its error is on molecules genuinely unlike the training set, what it does with an input outside its domain, and who checks it next month. If any answer is missing, the model is not ready — regardless of its R².
```

## Where the field is going

A short, honest view of the directions that look durable rather than merely loud:

- **Foundation models for chemistry and materials**, pretrained on large unlabelled corpora and fine-tuned on small measured sets — the direction suggested by the success of large interatomic-potential and structure models {cite:p}`merchant2023,jumper2021`. The open question is not capability but validation: how do you test a model whose training set you cannot enumerate?
- **Closed-loop autonomous laboratories** {cite:p}`burger2020,szymanski2023`, where the bottleneck shifts from prediction to characterisation and to deciding when a machine-made claim counts as a result.
- **Language models as laboratory interfaces** {cite:p}`bran2024` — reading procedures, writing instrument scripts, extracting data from the literature — with the extraction accuracy and provenance problems of Chapter 46 still unsolved.
- **Differentiable and physics-informed simulation** {cite:p}`karniadakis2021`, blurring the line between a simulator and a trained model, as in Chapter 50's hybrid.
- **Benchmarks and reporting standards**, the least glamorous and probably most consequential: shared, scaffold-split, uncertainty-aware benchmarks that make honest comparison possible, in the tradition of MoleculeNet {cite:p}`wu2018` and of open materials databases {cite:p}`jain2013,ong2013`.

The through-line of all five is the subject of this chapter. The field's limit is no longer how well a model can fit; it is how well we can tell when a model is fooling us.

## A closing word

You began this book, perhaps, by printing a number. You can now simulate a reactor, embed a molecule, train a graph network, run a Bayesian optimisation campaign, segment a micrograph, and — this chapter's point — say honestly how much any of it should be believed. That last skill is the rarest and the most valuable, and it is the one no library will do for you.

Go and measure something.

```{admonition} About the data and sources in this chapter
:class: note
All data are the simulated datasets described in Appendix B and generated by `tools/make_datasets.py`; the `pIC50_sim` values come from a structural rule applied by that script, not from any assay, and nothing here should be used to judge a real compound. Cited works are referred to for their ideas and conclusions only, with full details in the bibliography.
```

## Exercises

**★ 1.** Re-run the permutation importance with `scoring="accuracy"` instead of `"roc_auc"`. Do the rankings change? Explain why the choice of metric is part of the explanation.

**★ 2.** Raise and lower `GatedPredictor`'s `knn_cut` between the 90th and 99.9th percentiles of the training distances. Plot the fraction of test molecules answered against the accuracy on those answered. Where would you set it for a screening campaign, and where for a safety decision?

**★★ 3.** The leaky-feature demonstration used an obviously fake `batch_id`. Construct a *plausible* leak instead — for example a descriptor computed after the label was assigned — and check whether permutation importance alone would have caught it. What else would you need?

**★★ 4.** Write a `model_card()` function that takes a fitted pipeline, a test set and a dictionary of metadata and emits the JSON card above. Use it on two different chapters' models and compare the cards. Which fields were hard to fill in, and what does that tell you about those models?

**★★★ 5.** Implement conformal prediction for the classifier: hold out a calibration set, compute the non-conformity scores, and output prediction *sets* with a guaranteed marginal coverage of 90%. Compare the average set size on in-domain and out-of-domain inputs, and relate it to the risk–coverage curve above.

**★★★ 6.** Take any model you built earlier in this book and put it through the whole chapter: permutation importance, one local explanation, a calibration curve, an ensemble uncertainty, an OOD gate, a model card, and a pinned environment. Then write the two-paragraph limitations section you would be willing to publish. This is the capstone exercise of the book; nothing else you have learned matters as much as being able to do it.

```{admonition} Summary
:class: tip
- Trustworthiness is four answerable questions: why, how sure, is this in scope, and can I reproduce you.
- Permutation importance beats built-in importance; Shapley values explain single predictions and are plain arithmetic, not magic.
- An explanation can be faithful to the model and still reveal a disaster — it diagnoses leakage, it does not prevent it.
- Check calibration, separate epistemic from aleatoric uncertainty, and prove your uncertainty is useful with a risk–coverage curve.
- Models cannot refuse; wrap them in a pipeline that can, with an explicit applicability domain.
- FAIR data, a model card and a pinned environment are the minimum for work others can reuse; dual use, dataset bias and accountability are decisions to make before the loop closes, not after.
```
