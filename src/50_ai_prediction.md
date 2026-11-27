---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Predictive Chemistry: Reactions, Processes and Digital Twins

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 14, 15, 31, 33 and 34.
```

Everything so far in this Part has predicted a *property* of a *thing*: a solubility, a band gap, a gas class, a particle size. This chapter predicts the behaviour of a **process over time** — a reaction running in a flask, a column drifting out of specification, a catalyst slowly dying. That changes three things:

1. There is usually a **mechanistic model** already (rate laws, mass balances), and ignoring it is wasteful.
2. The data arrive as a **time series**, so random train/test splits are meaningless.
3. Someone will **act** on the prediction — raise a temperature, stop a batch, order a new catalyst charge — so an honest uncertainty matters more than a small gain in average error.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
rng = np.random.default_rng(50)
```

## Three ways to model a process

| Approach | What you supply | Strength | Weakness |
|---|---|---|---|
| **Mechanistic** (white box) | rate laws, balances, a few fitted constants | extrapolates; parameters mean something | wrong if the mechanism is wrong or incomplete |
| **Data-driven** (black box) | data only | captures whatever is really there, including the messy bits | interpolates only; needs lots of data; no physics |
| **Hybrid** (grey box) | the balances you trust + ML for the term you don't know | physics where you have it, learning where you don't | more work; you must decide *where* the ignorance sits |

The hybrid idea is old in chemical engineering — Psichogios and Ungar combined a neural network with first-principles balances in 1992 {cite:p}`psichogios1992`, and the approach has been reviewed repeatedly since {cite:p}`vonstosch2014`. It is now fashionable again under the name *physics-informed machine learning* {cite:p}`karniadakis2021`.

```{admonition} ⚗️ Chemistry first
:class: chem-note
The question is never "mechanistic or ML?" but "which part of my model am I actually ignorant about?". If you know the stoichiometry but not the rate law, learn the rate law. If you know the rate law but not how a catalyst deactivates with feed impurities, learn the deactivation. Learning things you already know wastes data and destroys extrapolation.
```

## A virtual reactor

Our running example is a batch reaction $A \xrightarrow{k_1} B \xrightarrow{k_2} C$, where $B$ is the product we want and $C$ is over-reaction waste. Both steps are Arrhenius in temperature. This simulator stands in for a real plant: we will *pretend* not to know parts of it.

```{code-cell} ipython3
R = 8.314e-3                       # kJ / (mol K)

# Prefactors are in min^-1, chosen so the chemistry is comfortable between 320 K and 370 K.
TRUE = dict(A1=3.5e8, Ea1=65.0,    # A -> B
            A2=1.9e12, Ea2=95.0)   # B -> C  (higher barrier: selectivity is temperature-sensitive)

def k_true(T, A, Ea):
    return A * np.exp(-Ea / (R * T))

def reactor(T_K, t_end=60.0, cA0=1.0, n=200, pars=TRUE):
    """Isothermal batch reactor. Returns t (min), concentrations (mol/L)."""
    k1 = k_true(T_K, pars["A1"], pars["Ea1"])          # per minute
    k2 = k_true(T_K, pars["A2"], pars["Ea2"])

    def rhs(t, y):
        cA, cB, cC = y
        return [-k1 * cA, k1 * cA - k2 * cB, k2 * cB]

    t = np.linspace(0, t_end, n)
    sol = solve_ivp(rhs, (0, t_end), [cA0, 0.0, 0.0], t_eval=t, rtol=1e-8, atol=1e-10)
    return t, sol.y

fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
for T in [320, 335, 350, 365]:
    t, y = reactor(T)
    axes[0].plot(t, y[1], label=f"{T} K")
axes[0].set(xlabel="time / min", ylabel="[B] / mol L$^{-1}$", title="product B: hotter is faster but less selective")
axes[0].legend(title="T")

Ts = np.linspace(315, 375, 60)
ybest = [reactor(T)[1][1].max() for T in Ts]
axes[1].plot(Ts, ybest, color="#e5487a", lw=2)
axes[1].axvline(Ts[int(np.argmax(ybest))], ls="--", c="grey")
axes[1].set(xlabel="T / K", ylabel="max [B] reached", title=f"optimum ≈ {Ts[int(np.argmax(ybest))]:.0f} K")
plt.tight_layout(); plt.show()
```

The maximum in the right panel is the whole point of process modelling: there is a best temperature, it is not at either end of the range, and it moves when anything else changes.

## Step 1: a pure data-driven surrogate

Suppose we run 40 experiments at random temperatures and quench times, measure $[B]$, and fit a model. This is what most "ML for reaction optimisation" work does.

```{code-cell} ipython3
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

def experiment(T_K, t_quench, noise=0.015, seed=None):
    """One lab run: quench at t_quench and assay B, with measurement noise."""
    r = np.random.default_rng(seed)
    t, y = reactor(T_K, t_end=max(t_quench, 1e-3), n=400)
    return float(np.clip(y[1][-1] + r.normal(0, noise), 0, None))

N = 60
T_lab = rng.uniform(325, 355, N)          # the range the lab actually explored
t_lab = rng.uniform(5, 50, N)
yB = np.array([experiment(T_lab[i], t_lab[i], seed=i) for i in range(N)])
lab = pd.DataFrame({"T_K": T_lab, "t_min": t_lab, "B": yB})
lab.head()
```

```{code-cell} ipython3
Xtr, Xte, ytr, yte = train_test_split(lab[["T_K", "t_min"]], lab["B"], test_size=0.3, random_state=0)
rf = RandomForestRegressor(n_estimators=400, random_state=0).fit(Xtr, ytr)
print(f"random-split test MAE   : {np.abs(rf.predict(Xte) - yte).mean():.4f} mol/L")

# now ask it about temperatures nobody ran
T_out = np.array([310.0, 318.0, 362.0, 372.0])
grid = pd.DataFrame({"T_K": T_out, "t_min": 25.0})
truth_out = np.array([experiment(T, 25.0, noise=0.0) for T in T_out])
print(f"extrapolation MAE      : {np.abs(rf.predict(grid) - truth_out).mean():.4f} mol/L")
pd.DataFrame({"T_K": T_out, "true B": truth_out.round(3), "RF": rf.predict(grid).round(3)})
```

The in-range error looks excellent and the extrapolation is nonsense — a random forest cannot leave the convex hull of its training data, it just repeats the nearest leaf value. A neural network would extrapolate differently but no more correctly.

## Step 2: hybrid modelling — keep the balances, learn the rate

Now assume what a chemist actually knows: the stoichiometry $A \to B \to C$ and that the rates are first order. We do **not** assume Arrhenius; we let the data tell us how $k_1$ and $k_2$ depend on temperature. The ODE solver enforces mass balance for free, so the model can never predict negative concentrations or more product than feed.

```{code-cell} ipython3
from scipy.optimize import curve_fit

def hybrid_predict(T_K, t_quench, k1, k2):
    """Known structure (A->B->C, first order), rate constants supplied from outside."""
    def rhs(t, y):
        cA, cB, cC = y
        return [-k1 * cA, k1 * cA - k2 * cB, k2 * cB]
    sol = solve_ivp(rhs, (0, max(t_quench, 1e-9)), [1.0, 0.0, 0.0], rtol=1e-8, atol=1e-10)
    return sol.y[1, -1]

# --- fit log k1, log k2 at each of several temperature bins -------------------
bins = np.array([327.5, 332.5, 337.5, 342.5, 347.5, 352.5])
rows = []
for Tc in bins:
    m = np.abs(lab["T_K"] - Tc) < 2.5
    if m.sum() < 4:
        continue
    tt, bb = lab.loc[m, "t_min"].to_numpy(), lab.loc[m, "B"].to_numpy()

    def model(t, lk1, lk2):
        return np.array([hybrid_predict(Tc, ti, np.exp(lk1), np.exp(lk2)) for ti in t])

    p, _ = curve_fit(model, tt, bb, p0=[np.log(0.05), np.log(0.01)],
                     bounds=([-9, -9], [2, 2]), maxfev=4000)
    rows.append(dict(T_K=Tc, n=int(m.sum()), k1=np.exp(p[0]), k2=np.exp(p[1])))

kfit = pd.DataFrame(rows)
kfit.assign(k1=lambda d: d.k1.round(4), k2=lambda d: d.k2.round(4))
```

Those fitted rate constants are *data*, not assumptions. Plotting $\ln k$ against $1/T$ is the classical test, and here it recovers the Arrhenius behaviour we hid in the simulator:

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(6.4, 4.4))
invT = 1000.0 / kfit["T_K"]
for col, c, lbl in [("k1", "#0f9d8a", "$k_1$ (A→B)"), ("k2", "#6c4ad6", "$k_2$ (B→C)")]:
    ax.plot(invT, np.log(kfit[col]), "o", color=c, label=lbl)
    cf = np.polyfit(invT, np.log(kfit[col]), 1)
    ax.plot(invT, np.polyval(cf, invT), "-", color=c, lw=1)
    Ea = -cf[0] * R * 1000.0
    print(f"{lbl}: fitted Ea = {Ea:6.1f} kJ/mol   (true {TRUE['Ea1'] if col=='k1' else TRUE['Ea2']:.1f})")
ax.set(xlabel="1000/T  /  K$^{-1}$", ylabel="ln k  (min$^{-1}$)", title="Arrhenius plot recovered from 60 runs")
ax.legend(); plt.tight_layout(); plt.show()
```

Look carefully at the two numbers. $E_{a,1}$ comes back almost exactly, because every run carries information about the first step. $E_{a,2}$ is noticeably off, because in the 325–355 K window most runs were quenched before much $B$ had turned into $C$, so the data say little about the second step. That is not a failure of the method; it is the data telling you which parameter it cannot pin down, which a black box would never tell you. Even so, we now have something the random forest could never give us: a model whose parameters extrapolate.

```{code-cell} ipython3
arr = {}
for col in ["k1", "k2"]:
    s, i = np.polyfit(1.0 / kfit["T_K"], np.log(kfit[col]), 1)
    arr[col] = (np.exp(i), -s * R)                      # (prefactor, Ea)

def hybrid(T_K, t_min):
    k1 = arr["k1"][0] * np.exp(-arr["k1"][1] / (R * T_K))
    k2 = arr["k2"][0] * np.exp(-arr["k2"][1] / (R * T_K))
    return hybrid_predict(T_K, t_min, k1, k2)

hy = np.array([hybrid(T, 25.0) for T in T_out])
cmp = pd.DataFrame({"T_K": T_out, "true B": truth_out, "RF": rf.predict(grid), "hybrid": hy})
print(cmp.round(3).to_string(index=False))
print(f"\nextrapolation MAE  RF {np.abs(cmp.RF - cmp['true B']).mean():.4f}"
      f"   hybrid {np.abs(cmp.hybrid - cmp['true B']).mean():.4f} mol/L")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(7.4, 4.6))
Tg = np.linspace(305, 378, 70)
ax.plot(Tg, [experiment(T, 25.0, noise=0.0) for T in Tg], "k-", lw=2, label="truth")
ax.plot(Tg, rf.predict(pd.DataFrame({"T_K": Tg, "t_min": 25.0})), color="#e5487a", label="random forest")
ax.plot(Tg, [hybrid(T, 25.0) for T in Tg], color="#0f9d8a", ls="--", lw=2, label="hybrid")
ax.axvspan(325, 355, color="#f2a93b", alpha=0.18, label="training range")
ax.set(xlabel="T / K", ylabel="[B] at 25 min", title="extrapolation is where physics pays")
ax.legend(); plt.tight_layout(); plt.show()
```

The hybrid is nearly exact at the cold end and drifts at the hot end. That residual error is the poorly determined $E_{a,2}$ from the Arrhenius plot showing up exactly where the second step matters most, so the model's weak spot is visible and explainable, and the fix is obvious: run two or three experiments hot and late.

```{admonition} 🤖 ML corner
:class: ml-corner
This is the single most useful idea in the chapter. **A black box is an excellent interpolator and a terrible extrapolator; a mechanism is the opposite.** So put the ML inside the mechanism, on the term you genuinely do not know, and let the conservation laws carry the extrapolation. In the example we used `curve_fit` plus an Arrhenius line because the unknown was one-dimensional; with ten unknown kinetic terms you would swap that line for a small neural network or a Gaussian process and keep everything else identical.
```

## Soft sensors: measuring what you cannot measure

Plants rarely have an on-line concentration meter. They have thermocouples, pressure transducers, flows, and perhaps a spectrometer. A **soft sensor** (inferential sensor) predicts the expensive laboratory value from the cheap on-line ones. Partial least squares is still the industry default {cite:p}`wold2001` because it tolerates hundreds of correlated spectral channels and few samples (Chapter 33).

```{code-cell} ipython3
# A simulated on-line NIR-like spectrum: three overlapping bands whose heights
# follow the three concentrations, plus baseline drift and detector noise.
wn = np.linspace(1100, 1800, 140)                     # "wavenumber" channels

def band(centre, width):
    return np.exp(-0.5 * ((wn - centre) / width) ** 2)

BASIS = np.vstack([band(1250, 45), band(1420, 60), band(1600, 50)])   # A, B, C

def spectrum(cA, cB, cC, r):
    s = np.array([cA, cB, cC]) @ BASIS
    s = s + 0.10 * (wn - wn.mean()) / np.ptp(wn) * r.uniform(-1, 1)     # sloping baseline
    return s + r.normal(0, 0.02, s.size)

r = np.random.default_rng(1)
recs, specs = [], []
for i in range(260):
    T = r.uniform(320, 360)
    t, y = reactor(T, t_end=60, n=120)
    j = r.integers(10, 120)
    cA, cB, cC = y[:, j]
    specs.append(spectrum(cA, cB, cC, r))
    recs.append(dict(T_K=T, t_min=t[j], cA=cA, cB=cB, cC=cC))
proc = pd.DataFrame(recs)
S = np.array(specs)
print(S.shape, proc.shape)

fig, ax = plt.subplots(figsize=(7.2, 3.8))
for i in range(6):
    ax.plot(wn, S[i], lw=1.2)
ax.set(xlabel="wavenumber / cm$^{-1}$", ylabel="absorbance", title="six simulated on-line spectra")
plt.tight_layout(); plt.show()
```

```{code-cell} ipython3
from sklearn.cross_decomposition import PLSRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

ntr = 180                                             # chronological split, not random
Str, Ste = S[:ntr], S[ntr:]
ytr_b, yte_b = proc["cB"][:ntr], proc["cB"][ntr:]

scores = {}
for nc in range(1, 9):
    pls = make_pipeline(StandardScaler(), PLSRegression(n_components=nc)).fit(Str, ytr_b)
    scores[nc] = r2_score(yte_b, pls.predict(Ste).ravel())
# parsimony rule: the smallest rank whose score is within 0.005 of the best
best = min(k for k, v in scores.items() if v >= max(scores.values()) - 0.005)
print("R² by number of PLS components:", {k: round(v, 3) for k, v in scores.items()})
print(f"best: {best} components, R² = {scores[best]:.3f}")

pls = make_pipeline(StandardScaler(), PLSRegression(n_components=best)).fit(Str, ytr_b)
pred_b = pls.predict(Ste).ravel()
rf_b = RandomForestRegressor(n_estimators=300, random_state=0).fit(Str, ytr_b)

fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.3))
axes[0].plot(sorted(scores), [scores[k] for k in sorted(scores)], "o-", color="#6c4ad6")
axes[0].set(xlabel="PLS components", ylabel="test R²", title="choose the rank, do not guess it")
axes[0].axvline(best, ls="--", c="grey")
axes[1].scatter(yte_b, pred_b, s=22, color="#0f9d8a", label=f"PLS  R²={scores[best]:.3f}")
axes[1].scatter(yte_b, rf_b.predict(Ste), s=16, color="#e5487a", alpha=.6,
                label=f"RF   R²={r2_score(yte_b, rf_b.predict(Ste)):.3f}")
lim = [0, max(yte_b.max(), pred_b.max()) * 1.05]
axes[1].plot(lim, lim, "k--", lw=1)
axes[1].set(xlabel="true [B]", ylabel="predicted [B]", title="soft sensor for [B] from spectra")
axes[1].legend(); plt.tight_layout(); plt.show()
```

The parsimony rule picks two components, which may look odd with three absorbing species. It is chemistry, not chance: in a closed batch $[A] + [B] + [C] = c_{A,0}$, so only two concentrations are independent and the spectra really do live in a two-dimensional space (plus noise and baseline). A third component buys almost nothing, and beyond that PLS is fitting drift. The rank of a good spectroscopic soft sensor is usually the number of *independent* chemical degrees of freedom, so mass balance shows up even in a regression.

```{admonition} ⚗️ Chemistry first
:class: chem-note
A soft sensor that works beautifully for a year and then fails is the normal outcome, because lamps age, windows foul and feedstock changes. That is the same drift problem as the electronic nose in Chapter 48, and the same fixes apply: periodic re-standardisation against lab samples, a monitored residual, and an alarm when the spectrum leaves the model's applicability domain.
```

## Forecasting with uncertainty: catalyst deactivation

A fixed-bed catalyst loses activity. The operator's real question is not "what is the activity now?" (a measurement answers that) but "when do I have to shut down and reload?", which is a forecast with a decision attached. We simulate 120 days of a slowly deactivating bed with weekly process upsets.

```{code-cell} ipython3
days = np.arange(0, 121)
activity = 1.0 * np.exp(-days / 700.0) - 0.0001 * days           # sintering + slow coking
activity += 0.012 * np.sin(2 * np.pi * days / 7)                  # weekly feed cycle
obs = activity + rng.normal(0, 0.008, days.size)
hist = pd.DataFrame({"day": days, "activity": obs})

SPEC = 0.75          # below this the unit is off specification
fig, ax = plt.subplots(figsize=(8.2, 4))
ax.plot(hist.day, hist.activity, ".", color="#6c4ad6", ms=5, label="daily activity")
ax.axhline(SPEC, color="#e5487a", ls="--", label=f"spec limit {SPEC}")
ax.set(xlabel="day on stream", ylabel="relative activity", title="120 days of history: when do we hit the limit?")
ax.legend(); plt.tight_layout(); plt.show()
```

A Gaussian process (Chapter 35, {cite:p}`rasmussen2006`) gives a forecast *with error bars*, which is exactly what a shutdown decision needs. Note the kernel choice: a long-scale RBF for the deactivation trend, an explicit periodic term for the weekly cycle, and a white-noise term for the measurement error.

```{code-cell} ipython3
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ExpSineSquared, ConstantKernel as C

X = hist[["day"]].to_numpy(float)
y = hist["activity"].to_numpy()

kern = (C(1.0, (1e-3, 1e3)) * RBF(80.0, (10.0, 1e4))
        + C(0.02, (1e-5, 1.0)) * ExpSineSquared(length_scale=10.0, periodicity=7.0,
                                                periodicity_bounds="fixed")
        + WhiteKernel(1e-4, (1e-8, 1e-1)))
gp = GaussianProcessRegressor(kernel=kern, normalize_y=True, n_restarts_optimizer=2,
                              random_state=0).fit(X, y)
print(gp.kernel_)

fut = np.arange(0, 241).reshape(-1, 1)
mu, sd = gp.predict(fut, return_std=True)

fig, ax = plt.subplots(figsize=(9, 4.6))
ax.plot(X, y, ".", color="#6c4ad6", ms=5, label="observed")
ax.plot(fut, mu, color="#0f9d8a", lw=2, label="GP mean forecast")
ax.fill_between(fut.ravel(), mu - 2 * sd, mu + 2 * sd, color="#0f9d8a", alpha=0.20, label="±2σ")
ax.axhline(SPEC, color="#e5487a", ls="--")
ax.axvline(120, color="grey", ls=":")
ax.set(xlabel="day on stream", ylabel="relative activity", ylim=(0.55, 1.08),
       title="forecast to 240 days")
ax.legend(loc="lower left"); plt.tight_layout(); plt.show()
```

The band widens as we forecast further, which is the honest answer. Turn it into the number the operator wants — the distribution of the crossing day — by sampling from the posterior:

```{code-cell} ipython3
samples = gp.sample_y(fut, n_samples=400, random_state=0)       # (n_days, n_samples)
cross = []
for s in samples.T:
    below = np.where(s < SPEC)[0]
    if below.size:
        cross.append(fut.ravel()[below[0]])
cross = np.array(cross)
print(f"{len(cross)}/400 sampled trajectories cross the limit within 240 days")
print(f"median crossing day : {np.median(cross):.0f}")
print(f"10th–90th percentile: {np.percentile(cross, 10):.0f} – {np.percentile(cross, 90):.0f}")

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.hist(cross, bins=28, color="#f2a93b", edgecolor="w")
ax.axvline(np.median(cross), color="#e5487a", lw=2)
ax.set(xlabel="day the spec limit is first crossed", ylabel="count",
       title="a maintenance planner can use this")
plt.tight_layout(); plt.show()
```

### Validate forecasts the way you will use them

Never random-split a time series. The right scheme is **rolling origin**: fit on the first $n$ points, predict a fixed horizon ahead, step forward, repeat. Severson and co-workers made the same point for battery life, predicting end of life from only the earliest cycles and validating on later cells {cite:p}`severson2019`.

```{code-cell} ipython3
from sklearn.linear_model import LinearRegression

HORIZON = 20
rows = []
for origin in range(40, 101, 10):
    Xa, ya = X[:origin], y[:origin]
    Xb = X[origin:origin + HORIZON]
    yb = y[origin:origin + HORIZON]
    if len(yb) < HORIZON:
        break
    g = GaussianProcessRegressor(kernel=kern, normalize_y=True, random_state=0).fit(Xa, ya)
    mu_b, sd_b = g.predict(Xb, return_std=True)
    lin = LinearRegression().fit(Xa, ya)
    inside = np.mean(np.abs(yb - mu_b) <= 2 * sd_b)
    rows.append(dict(origin=origin,
                     gp_mae=np.abs(mu_b - yb).mean(),
                     linear_mae=np.abs(lin.predict(Xb) - yb).mean(),
                     coverage_2sigma=inside))
roll = pd.DataFrame(rows)
print(roll.round(4).to_string(index=False))
print(f"\nmean 20-day MAE   GP {roll.gp_mae.mean():.4f}   linear {roll.linear_mae.mean():.4f}")
print(f"mean ±2σ coverage (want ≈0.95): {roll.coverage_2sigma.mean():.2f}")
```

Two things to read off. The GP beats a straight line because deactivation is curved, and the coverage tells you whether the error bars are *calibrated* — a model whose 95% interval contains 60% of future points is dangerous however good its MAE looks.

## A small digital twin

A "digital twin" is a model of a specific unit that is kept in step with that unit by live data and used to answer what-if questions. Nothing magical: a simulator, a state estimate, a forecaster, and a decision rule, wired together. Here is a minimal one in about forty lines.

```{code-cell} ipython3
class ReactorTwin:
    """Mechanistic model + soft sensor + drift tracking + a setpoint recommendation."""

    def __init__(self, arrhenius, soft_sensor, spec=0.75):
        self.arr, self.soft, self.spec = arrhenius, soft_sensor, spec
        self.health = 1.0           # multiplies k1: our estimate of catalyst activity
        self.log = []

    # --- mechanistic prediction, scaled by current health --------------------
    def predict_B(self, T_K, t_min, health=None):
        h = self.health if health is None else health
        k1 = h * self.arr["k1"][0] * np.exp(-self.arr["k1"][1] / (R * T_K))
        k2 = self.arr["k2"][0] * np.exp(-self.arr["k2"][1] / (R * T_K))
        return hybrid_predict(T_K, t_min, k1, k2)

    # --- assimilate one batch: measured spectrum -> [B] -> update health -----
    def update(self, spectrum_row, T_K, t_min, lr=0.35):
        measured = float(self.soft.predict(spectrum_row.reshape(1, -1)).ravel()[0])
        expected = self.predict_B(T_K, t_min)
        if expected > 1e-6:
            ratio = np.clip(measured / expected, 0.2, 2.0)
            self.health *= (1 - lr) + lr * ratio          # gentle multiplicative update
        self.log.append(dict(T_K=T_K, t_min=t_min, measured=measured,
                             expected=expected, health=self.health))
        return measured

    # --- what-if: best temperature for a fixed batch time -------------------
    def best_T(self, t_min, lo=315.0, hi=375.0, n=50):
        Tg = np.linspace(lo, hi, n)
        vals = [self.predict_B(T, t_min) for T in Tg]
        return float(Tg[int(np.argmax(vals))]), float(np.max(vals))

    def status(self):
        return ("OK" if self.health >= self.spec
                else "ACTION: activity below spec, recommend regeneration")
```

Now run it against the "real" plant for 60 batches, while the plant's catalyst quietly decays — something the twin was never told about.

```{code-cell} ipython3
twin = ReactorTwin(arr, pls)
r2 = np.random.default_rng(99)
T_set, t_batch = 345.0, 25.0
hist_rows = []

for b in range(60):
    true_health = np.exp(-b / 70.0)                    # the plant decays; the twin must notice
    pars = dict(TRUE); pars["A1"] = TRUE["A1"] * true_health
    t, ycur = reactor(T_set, t_end=t_batch, n=200, pars=pars)
    cA, cB, cC = ycur[:, -1]
    spec_row = spectrum(cA, cB, cC, r2)
    meas = twin.update(spec_row, T_set, t_batch)
    # once the twin thinks activity has dropped, it re-optimises the setpoint
    if b >= 20:
        T_set, _ = twin.best_T(t_batch)
    hist_rows.append(dict(batch=b, true_health=true_health, est_health=twin.health,
                          B_true=cB, B_soft=meas, T_set=T_set))

tw = pd.DataFrame(hist_rows)
print(tw.iloc[::10].round(3).to_string(index=False))
print("\n" + twin.status())
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
axes[0].plot(tw.batch, tw.true_health, "k-", lw=2, label="true activity")
axes[0].plot(tw.batch, tw.est_health, color="#0f9d8a", ls="--", lw=2, label="twin estimate")
axes[0].set(xlabel="batch", ylabel="relative activity", title="state estimation")
axes[0].legend()

axes[1].plot(tw.batch, tw.B_true, "k-", lw=2, label="true [B]")
axes[1].plot(tw.batch, tw.B_soft, ".", color="#6c4ad6", label="soft sensor")
axes[1].set(xlabel="batch", ylabel="[B] / mol L$^{-1}$", title="product, as the bed ages")
axes[1].legend()

axes[2].plot(tw.batch, tw.T_set, color="#e5487a", lw=2)
axes[2].axvline(20, ls=":", c="grey")
axes[2].set(xlabel="batch", ylabel="setpoint / K", title="twin raises T to defend yield")
plt.tight_layout(); plt.show()

err = np.abs(tw.est_health - tw.true_health)
print(f"mean |health error| after batch 10: {err[10:].mean():.3f}")
```

The twin tracks the decay it was never told about, because the discrepancy between the mechanistic expectation and the soft-sensor reading has nowhere else to go. That is the generic trick for any grey-box monitor: **one unexplained residual, one lumped parameter to absorb it.**

```{admonition} Read your own results honestly
:class: warning
Our twin succeeds partly because we cheated in a way real plants do not allow: there is exactly *one* unknown (activity) and the soft sensor was trained on the same simulator. On a real unit, fouling, a thermocouple bias and a feed composition change all push the same residual, and the estimator will happily blame the wrong one. The defences are instrumentation (measure more than one thing), identifiability analysis before you deploy (can these parameters even be separated by this data?), and the discipline of reporting an estimate with its confidence interval. A digital twin that outputs a single number with no uncertainty is a dashboard, not a model.
```

## Where predictive chemistry is going

Four directions worth watching, all documented in the open literature:

- **Reaction outcome prediction** from text-like representations of reactions, treating synthesis as a translation problem {cite:p}`schwaller2019`, and its complement, computer-assisted retrosynthesis {cite:p}`segler2018`.
- **Bayesian optimisation of real reactions**, where the model's uncertainty chooses the next experiment rather than merely predicting {cite:p}`shields2021` — the ideas of Chapter 47 applied to a running process.
- **Lifetime and failure forecasting** from early-life data, exemplified by battery cycle-life prediction {cite:p}`severson2019`; the same framing fits catalyst runs, membrane fouling and stability testing.
- **Physics-informed networks and differentiable simulators** {cite:p}`karniadakis2021`, which let you put a neural network *inside* an ODE solver and train through it, generalising this chapter's hand-built hybrid.

```{admonition} About the data in this chapter
:class: note
Every number here comes from the simulators defined in the cells above — the batch reactor, the synthetic spectra and the deactivation series. No experimental dataset is reproduced. The cited papers are referred to for their ideas and results only; read them at their DOIs.
```

## Exercises

**★ 1.** Re-run the virtual reactor with `TRUE["Ea2"]` lowered to 70 kJ/mol so the two barriers are similar. What happens to the optimal temperature curve in the first figure, and why?

**★ 2.** Change the soft-sensor split from chronological to random (`train_test_split`) and compare the reported R². Which number would you put in a report, and what would you write beside it?

**★★ 3.** The hybrid model assumed first-order kinetics. Break that assumption: make the first step second order in A inside `reactor`, keep the first-order `hybrid_predict`, refit, and plot the Arrhenius fit. How does a wrong mechanism show up — as bad fits, or as good fits with meaningless parameters?

**★★ 4.** Add a second unknown to the digital twin: let the plant's `A2` also drift, and give `ReactorTwin` a second health parameter for it. Can the twin separate the two from one measured concentration? Explain your answer in terms of identifiability.

**★★★ 5.** Replace the GP forecast with a mechanistic deactivation law, $a(t) = a_0 e^{-t/\tau}$, fitted by `curve_fit` with its covariance matrix, and propagate the parameter uncertainty into the crossing-day distribution. Compare the resulting interval with the GP's. Which is wider, and which do you trust more 200 days out?

**★★★ 6.** Build a rolling-origin evaluation for the *soft sensor* instead of the forecaster: retrain on the first $n$ batches, test on the next 20, and plot test R² against $n$ for both PLS and the random forest. At what point, if ever, does retraining stop helping?

```{admonition} Summary
:class: tip
- Process prediction differs from property prediction: there is physics to reuse, the data are ordered in time, and a decision follows.
- Black-box models interpolate well and extrapolate catastrophically. Put the ML on the term you are ignorant about and let the balances do the rest.
- Soft sensors infer expensive laboratory values from cheap on-line signals; PLS remains the workhorse, and its optimal rank often equals the number of absorbing species.
- Forecasts need calibrated uncertainty, validated by rolling origin, and are most useful when converted into the distribution of a decision variable such as the shutdown day.
- A digital twin is a simulator plus a state estimate plus a decision rule. Its weakness is identifiability: one residual cannot be blamed on three causes.
```
