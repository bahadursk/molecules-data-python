---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Biochemistry and Environmental Chemistry

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 23, 25 and Part V.
```

Two fields where chemistry meets life and the planet, and where machine learning has become everyday practice: protein structure and function {cite:p}`jumper2021`, and the prediction of how chemicals behave in air, water and organisms. In biochemistry the central objects are proteins, chains of amino acids whose sequence decides how they fold and what they do, and the small molecules that bind to them. Reading a protein's sequence is now routine, but measuring its structure, its function or how tightly a molecule binds to it is slow, so models that predict those from sequence or from simple descriptors are everywhere. In environmental chemistry the question is what becomes of a chemical after it is released: whether it stays in water or collects in fatty tissue, how long it survives, and whether it is harmful. Regulators must form a view on tens of thousands of substances, most never tested, so they lean on **QSPR** and **QSAR** models (Chapter 36) that estimate these properties from structure. The chapter covers sequence features, a hydropathy plot, a binding model, a hybrid treatment of enzyme kinetics, environmental-fate screening and structural alerts.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Draw
rng = np.random.default_rng(0)

DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name, **kw):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name, **kw)
```

## Proteins as sequences

A protein is a string over a 20-letter alphabet. Simple sequence features already predict a great deal. Each letter stands for one amino acid, and each amino acid has known properties. The **Kyte–Doolittle** scale gives every amino acid a hydropathy value, positive for water-avoiding residues such as isoleucine and leucine, negative for water-loving ones such as arginine and aspartate; its average over a sequence is called **GRAVY**, the grand average of hydropathy, and it is high for proteins that sit in membranes. Aspartate and glutamate carry a negative charge near pH 7, lysine and arginine a positive one, so the net charge follows from counting letters. Python strings count and iterate directly (Chapter 3), so each feature is a single expression:

```{code-cell} ipython3
pep = "MKLLVAGDE"
print(pep.count("L") / len(pep), sum(a in "DE" for a in pep), sum(a in "KR" for a in pep))
```

The amino-acid **composition**, the fraction of each of the twenty letters, supplies twenty more features. The three families below are synthetic, generated with different composition biases: one rich in hydrophobic residues, one in charged and polar ones, and one in glycine and proline.

```{code-cell} ipython3
AA = "ACDEFGHIKLMNPQRSTVWY"
KD = dict(zip(AA, [1.8, 2.5, -3.5, -3.5, 2.8, -0.4, -3.2, 4.5, -3.9, 3.8, 1.9, -3.5, -1.6, -3.5, -4.5, -0.8, -0.7, 4.2, -0.9, -1.3]))
CHARGE = {"D": -1, "E": -1, "K": 1, "R": 1, "H": 0.1}
MASS = dict(zip(AA, [71.08, 103.14, 115.09, 129.12, 147.18, 57.05, 137.14, 113.16, 128.17, 113.16, 131.19,
                     114.10, 97.12, 128.13, 156.19, 87.08, 101.10, 99.13, 186.21, 163.18]))

def sequence_features(seq):
    n = len(seq)
    comp = {f"frac_{a}": seq.count(a) / n for a in AA}
    return {**comp, "length": n, "mass_kDa": sum(MASS[a] for a in seq) / 1000,
            "gravy": np.mean([KD[a] for a in seq]), "net_charge": sum(CHARGE.get(a, 0) for a in seq),
            "frac_charged": sum(a in "DEKR" for a in seq) / n,
            "frac_hydrophobic": sum(a in "AVLIMFWC" for a in seq) / n,
            "frac_aromatic": sum(a in "FWY" for a in seq) / n}

# Three synthetic protein families with different composition biases
FAMILIES = {"membrane": "AVLIMFWG", "soluble enzyme": "DEKRSTNQGHY", "structural": "GPAVKRE"}
rows = []
for fam, alphabet in FAMILIES.items():
    for _ in range(140):
        n = rng.integers(120, 400)
        seq = "".join(rng.choice(list(alphabet + AA), n, p=np.r_[np.full(len(alphabet), 0.8 / len(alphabet)),
                                                                 np.full(20, 0.2 / 20)]))
        rows.append({"family": fam, "seq": seq, **sequence_features(seq)})
prot = pd.DataFrame(rows)
prot[["family", "length", "gravy", "net_charge", "frac_aromatic"]].groupby("family").mean().round(2)
```

```{code-cell} ipython3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.decomposition import PCA

feat_cols = [c for c in prot.columns if c not in ("family", "seq")]
Xp = prot[feat_cols].values
pred = cross_val_predict(RandomForestClassifier(300, n_jobs=-1, random_state=0), Xp, prot.family,
                         cv=StratifiedKFold(5, shuffle=True, random_state=0))
print(classification_report(prot.family, pred))
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
T = PCA(2).fit_transform((Xp - Xp.mean(0)) / Xp.std(0))
for fam, c in zip(FAMILIES, ["#6c4ad6", "#0f9d8a", "#e5487a"]):
    m = (prot.family == fam).values
    axes[0].scatter(T[m, 0], T[m, 1], s=14, color=c, label=fam)
axes[0].legend(); axes[0].set(xlabel="PC1", ylabel="PC2", title="Sequence-composition space")
ConfusionMatrixDisplay.from_predictions(prot.family, pred, ax=axes[1], cmap="Purples", colorbar=False, xticks_rotation=30)
plt.tight_layout(); plt.show()
```

### A hydropathy plot: finding membrane helices

A protein that crosses a lipid membrane does so with a stretch of roughly twenty hydrophobic residues wound into a helix. A **hydropathy plot** looks for those stretches by averaging the Kyte–Doolittle values over a sliding window of about nineteen residues; where the average rises above roughly 1.6, a membrane-spanning segment is likely. The moving average is a convolution with a flat window:

```{code-cell} ipython3
print(np.convolve([1, 2, 3, 4, 5], np.ones(3) / 3, mode="valid"))
```

Our sequences are random apart from their composition, so the peaks below are scattered rather than forming the clean blocks of a real membrane protein, but the calculation is the one used on real sequences.

```{code-cell} ipython3
seq = prot.query("family == 'membrane'").seq.iloc[0][:220]
window = 19
h = np.convolve([KD[a] for a in seq], np.ones(window) / window, mode="valid")
x = np.arange(len(h)) + window // 2
fig, ax = plt.subplots(figsize=(9, 3.2))
ax.plot(x, h, color="#6c4ad6")
ax.fill_between(x, h, 1.6, where=h > 1.6, color="#f2a93b", alpha=0.6)
ax.axhline(1.6, ls="--", color="k")
ax.set(xlabel="residue", ylabel="mean Kyte–Doolittle hydropathy", title="Peaks above 1.6 suggest transmembrane segments")
plt.show()
```

## Protein–ligand binding

Binding affinity depends on both partners. A small molecule binds in a **pocket**, a cavity on the protein's surface, and how tightly depends on how well the two match in shape and in polarity: a greasy molecule fits a greasy pocket, a hydrogen-bond donor needs an acceptor to meet. A model of binding therefore needs features of both partners, and the simplest featurisation concatenates ligand descriptors with pocket descriptors. Here each pocket is summarised by three numbers, its preferred lipophilicity, how polar it is and its volume, and the simulated score rewards matching the ligand's logP and size to the pocket's, plus terms for hydrogen bonding in polar pockets and aromatic rings in greasy ones. The score is defined so that lower is better, as docking programs report it. The Gaussian terms give their maximum when ligand and pocket agree:

```{code-cell} ipython3
for lp in [1.0, 3.5, 6.0]:
    print(lp, round(float(np.exp(-((lp - 3.5) / 2.2) ** 2)), 2))   # best match at the pocket's 3.5
```

```{code-cell} ipython3
LIGANDS = ["CC(=O)Oc1ccccc1C(=O)O", "CC(C)Cc1ccc(cc1)C(C)C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
           "Nc1ccc(cc1)S(=O)(=O)N", "OC(=O)c1ccccc1O", "CC(=O)Nc1ccc(O)cc1", "c1ccc2[nH]c(nc2c1)C(=O)N",
           "COc1ccc2cc(ccc2c1)C(C)C(=O)O", "CCN(CC)CCOC(=O)c1ccc(N)cc1", "Cc1ccccc1NC(=O)CN(CC)CC",
           "CC(C)(C)NCC(O)c1ccc(O)c(O)c1", "Cn1cnc2c1c(=O)[nH]c(=O)n2C", "OCC(O)C(O)C(O)C(O)CO",
           "c1ccc(cc1)C(=O)Nc1ccccc1", "CC(=O)c1ccc(N)cc1", "NC(=O)c1ccncc1"]
POCKETS = {"hydrophobic deep": (3.5, 0.2, 520), "polar shallow": (0.5, 0.8, 260),
           "charged": (-0.5, 0.9, 320), "aromatic box": (2.5, 0.3, 400)}

rows = []
for smi in LIGANDS:
    m = Chem.MolFromSmiles(smi)
    lig = {"MW": Descriptors.MolWt(m), "logP": Crippen.MolLogP(m), "TPSA": rdMolDescriptors.CalcTPSA(m),
           "HBD": rdMolDescriptors.CalcNumHBD(m), "arom": rdMolDescriptors.CalcNumAromaticRings(m)}
    for pocket, (p_logp, p_polar, p_vol) in POCKETS.items():
        # A transparent "docking score": shape and polarity complementarity, plus noise
        fit = -(2.0 * np.exp(-((lig["logP"] - p_logp) / 2.2) ** 2)
                + 1.2 * np.exp(-((lig["MW"] - p_vol * 0.7) / 120) ** 2)
                + 0.8 * p_polar * lig["HBD"] / 3 + 0.4 * (1 - p_polar) * lig["arom"])
        rows.append({"ligand": smi, "pocket": pocket, **lig, "p_logp": p_logp, "p_polar": p_polar, "p_vol": p_vol,
                     "score_sim": fit + rng.normal(0, 0.15)})
bind = pd.DataFrame(rows)
piv = bind.pivot(index="ligand", columns="pocket", values="score_sim")
plt.figure(figsize=(7, 6))
sns.heatmap(piv, cmap="RdYlGn_r", annot=True, fmt=".1f", yticklabels=[s[:26] for s in piv.index],
            cbar_kws={"label": "simulated score (lower = better)"})
plt.title("Ligand × pocket selectivity matrix"); plt.tight_layout(); plt.show()
```

Every ligand appears with every pocket, so the data can be split in two quite different ways with `GroupKFold` (Chapter 33). Grouping by ligand asks how well the model scores a new molecule against pockets it knows. Grouping by pocket asks how well it scores known molecules against a protein it has never seen, which is the situation whenever a project moves to a new target.

```{code-cell} ipython3
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

X = bind[["MW", "logP", "TPSA", "HBD", "arom", "p_logp", "p_polar", "p_vol"]].values
y = bind.score_sim.values
p_lig = cross_val_predict(GradientBoostingRegressor(random_state=0), X, y, cv=GroupKFold(4), groups=bind.ligand)
p_pock = cross_val_predict(GradientBoostingRegressor(random_state=0), X, y, cv=GroupKFold(4), groups=bind.pocket)
print(f"unseen ligands R² = {r2_score(y, p_lig):.2f}   unseen pockets R² = {r2_score(y, p_pock):.2f}")
```

Generalising to a **new target** is much harder than to a new ligand on a known target, the central difficulty of structure-based affinity prediction.

## Enzyme kinetics meets ML

A hybrid approach: fit mechanistic Michaelis–Menten parameters per variant, then learn how sequence changes shift those parameters. An enzyme's rate rises with substrate concentration and then levels off, and the Michaelis–Menten equation $v = V_\max [S]/(K_M + [S])$ (Chapter 25) describes that with two numbers: $V_\max$, the rate when every enzyme molecule is occupied, and $K_M$, the concentration at which the rate reaches half of $V_\max$, which measures how tightly the substrate is held. A mutation can change either one, and the ratio $k_\mathrm{cat}/K_M$, here approximated by $V_\max/K_M$, measures the overall efficiency at low substrate concentration. Fitting the two parameters to measured rates is a non-linear least-squares problem for `curve_fit` (Chapter 16). The **Lineweaver–Burk** plot of $1/v$ against $1/[S]$ turns the curve into a straight line, which is how the parameters were read off before computers:

```{code-cell} ipython3
S_demo = np.array([1.0, 5.0, 20.0])
print((12.0 * S_demo / (4.0 + S_demo)).round(2), "µM/s at Vmax = 12, KM = 4")
```

Fitting mechanism first and modelling only the parameters is a useful pattern in general: it keeps the physics explicit and leaves the model a much smaller job.

```{code-cell} ipython3
from scipy.optimize import curve_fit

def mm(S, Vmax, KM):
    return Vmax * S / (KM + S)

S = np.array([0.2, 0.5, 1, 2, 5, 10, 20, 50])
variants = {"wild type": (12.0, 4.0), "A54V": (11.0, 2.5), "D88N": (4.0, 4.5),
            "K120R": (13.5, 6.0), "triple": (3.0, 1.2)}
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
fitted = {}
for (name, (Vmax, KM)), c in zip(variants.items(), plt.cm.viridis(np.linspace(0, 0.85, 5))):
    v = mm(S, Vmax, KM) * (1 + rng.normal(0, 0.05, S.size))
    popt, _ = curve_fit(mm, S, v, p0=[10, 3])
    fitted[name] = popt
    Sf = np.linspace(0, 55, 200)
    axes[0].plot(Sf, mm(Sf, *popt), color=c); axes[0].plot(S, v, "o", color=c, ms=5, label=name)
    axes[1].plot(1 / S, 1 / v, "o-", color=c, ms=4)
axes[0].legend(fontsize=8); axes[0].set(xlabel="[S] / mM", ylabel="v / μM s$^{-1}$", title="Michaelis–Menten fits")
axes[1].set(xlabel="1/[S]", ylabel="1/v", title="Lineweaver–Burk view")
plt.tight_layout(); plt.show()
pd.DataFrame(fitted, index=["Vmax", "KM"]).T.assign(kcat_over_KM=lambda d: d.Vmax / d.KM).round(2)
```

`D88N` loses turnover (a catalytic residue) while `A54V` improves binding: the kind of structure–function story that sequence-to-activity models try to learn at scale.

## Environmental fate and toxicity

Three quantities govern what happens to a chemical released into the environment: how it partitions (log K_ow), how long it persists, and how much it accumulates in organisms. The **octanol–water partition coefficient** $K_\mathrm{ow}$ is the ratio of a chemical's concentration in octanol, a stand-in for fat, to that in water; its logarithm is the logP of Chapter 20. A high value means the chemical prefers fatty tissue, so it collects in organisms, measured as the **bioconcentration factor** (BCF), the ratio of its concentration in an organism to that in the surrounding water. **Persistence** is usually reported as a half-life in soil or water, and carbon–halogen bonds resist the enzymes that break molecules down, so halogenated compounds last longest. A substance that is persistent, bioaccumulative and toxic at once is flagged as **PBT** and restricted. The relations below are deliberately simple linear forms in the spirit of regulatory screening models, with simulated inputs:

```{code-cell} ipython3
print(round(0.85 * 5.0 - 0.70, 2), "log BCF for a chemical with logKow = 5")
```

```{code-cell} ipython3
mols = load_data("molecules_sim.csv")
m_objs = [Chem.MolFromSmiles(s) for s in mols.smiles]

# Transparent QSPR-style relations, as used in regulatory screening models
logKow = mols.logP
BCF = 0.85 * logKow - 0.70                                      # bioconcentration factor (log units)
halflife = 10 ** (0.35 * logKow + 0.2 * mols.smiles.str.count("Cl|Br|F").astype(float) + 1.0)   # days in soil
mobility = -0.6 * logKow + 2.0                                   # log Koc proxy: low Kow = mobile in water

env = pd.DataFrame({"smiles": mols.smiles, "logKow": logKow, "logBCF": BCF, "halflife_d": halflife, "log_mobility": mobility,
                    "halogens": mols.smiles.str.count("Cl|Br|F")})
env["PBT_flag"] = ((env.logBCF > 3.3) & (env.halflife_d > 120)).astype(int)
print(f"{env.PBT_flag.sum()} of {len(env)} molecules flagged as potentially persistent and bioaccumulative")

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
sc = axes[0].scatter(env.logKow, env.logBCF, c=env.halflife_d, cmap="inferno_r", s=10, norm="log")
axes[0].axhline(3.3, ls="--", color="k"); plt.colorbar(sc, ax=axes[0], label="half-life / d")
axes[0].set(xlabel="log K$_{ow}$", ylabel="log BCF", title="Bioaccumulation rises with lipophilicity")
sns.boxplot(env, x="halogens", y="halflife_d", hue="halogens", legend=False, ax=axes[1], palette="YlOrRd", log_scale=(False, True))
axes[1].set(title="Halogens slow degradation", ylabel="half-life / d")
axes[2].scatter(env.log_mobility, env.logBCF, c=env.PBT_flag, cmap="coolwarm", s=10)
axes[2].set(xlabel="mobility in water (log)", ylabel="log BCF", title="The regulator's trade-off: mobile or accumulating")
plt.tight_layout(); plt.show()
```

### A structural-alert toxicity model

A **structural alert** is a substructure associated with a hazard, such as a nitroaromatic group or an epoxide, and lists of them are used in regulatory screening. A model that sees substructures can learn such rules, whereas one given only bulk properties such as molecular weight and logP cannot. The comparison below trains the same classifier on descriptors, on Morgan fingerprints and on both, and reports the **ROC-AUC** (Chapter 31): the probability that a randomly chosen toxic molecule is ranked above a randomly chosen non-toxic one, where 0.5 is chance. `class_weight="balanced"` compensates for there being far fewer toxic molecules than safe ones.

```{code-cell} ipython3
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from rdkit.Chem import rdFingerprintGenerator

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
Xf = np.array([gen.GetFingerprintAsNumPy(m) for m in m_objs])
Xd = mols[["MW", "logP", "TPSA", "HBD", "RotB"]].values
cv = StratifiedKFold(5, shuffle=True, random_state=0)
fig, ax = plt.subplots(figsize=(5.5, 5))
for X, name, c in [(Xd, "descriptors", "#e5487a"), (Xf, "fingerprints", "#0f9d8a"),
                   (np.hstack([Xd, Xf]), "both", "#6c4ad6")]:
    p = cross_val_predict(RandomForestClassifier(300, class_weight="balanced", n_jobs=-1, random_state=0), X,
                          mols.toxic_sim, cv=cv, method="predict_proba")[:, 1]
    print(f"{name:14s} ROC-AUC = {roc_auc_score(mols.toxic_sim, p):.3f}")
    RocCurveDisplay.from_predictions(mols.toxic_sim, p, ax=ax, name=f"{name} (AUC {roc_auc_score(mols.toxic_sim, p):.2f})",
                                     curve_kwargs={"color": c})
ax.plot([0, 1], [0, 1], "k:"); ax.set_title("Toxicity: substructures help, but not by much"); plt.show()
```

Fingerprints, which can see substructures, do better than bulk descriptors (ROC-AUC about 0.64 against 0.60), and combining the two is best, but none of the three is close to accurate. That is a fair reflection of real toxicity prediction: the signal is genuinely there, since substructures help, but a single label per molecule for a property that depends on dose, route and species is a hard target. Regulatory tools therefore combine alerts, statistical models and expert judgement rather than relying on any one.

```{code-cell} ipython3
alerts = {"nitroaromatic": "c[N+](=O)[O-]", "aromatic amine": "c[NX3;H2]", "Michael acceptor": "C=CC=O",
          "epoxide": "C1OC1", "aryl halide": "c[Cl,Br]", "aldehyde": "[CX3H1]=O"}
hits = {k: sum(m.HasSubstructMatch(Chem.MolFromSmarts(v)) for m in m_objs) for k, v in alerts.items()}
pd.Series(hits).sort_values().plot.barh(color="#f2a93b", figsize=(6.5, 3)); plt.xlabel("molecules matching")
plt.title("Structural alerts in the library"); plt.show()
```

```{warning}
Structural alerts are hypotheses, not verdicts. Many safe drugs contain alerting groups, and a model trained on alerts will reproduce their biases. Regulatory use requires validated models, documented applicability domains and expert review.
```

```{admonition} Summary
:class: tip
- Amino-acid composition, GRAVY and net charge are cheap sequence features that already classify protein families.
- A sliding-window hydropathy plot locates candidate membrane-spanning segments.
- Binding models need features of both ligand and pocket; generalising to a new target is far harder than to a new ligand.
- Fit mechanistic parameters such as $V_\max$ and $K_M$ first, then model how they change.
- Environmental screening combines partitioning, persistence and bioaccumulation into PBT flags.
- Fingerprints beat bulk descriptors for alert-driven toxicity; alerts are hypotheses, not verdicts.
```

## Exercises

**★ 1.** Add a "frac_proline" feature to the protein model. Which family does it help identify?

**★★ 2.** Fit the enzyme variants' kcat/KM and plot it against the number of mutations.

**★★★ 3.** Build a simple multimedia fate model: given log K_ow and half-life, estimate the fraction of a chemical in air, water, soil and biota at steady state.
