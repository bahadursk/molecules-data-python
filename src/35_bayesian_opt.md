---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Bayesian Optimisation and Active Learning

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 16 and 31.
```

Experiments are expensive. When each reaction takes a day, you want to find the best conditions in as few runs as possible. **Bayesian optimisation** (BO) builds a probabilistic model of the response, then chooses the next experiment by balancing *exploitation* (go where the model predicts good results) and *exploration* (go where the model is unsure). It is the brain inside many self-driving laboratories. The word *Bayesian* refers to the way the model is updated: it starts with broad beliefs about the response, and each new measurement narrows those beliefs where it was made. Compared with a grid search, which spends experiments evenly everywhere, or changing one variable at a time, which can miss interactions between variables, BO usually reaches a good optimum in far fewer runs. This chapter builds the two ingredients, a Gaussian-process model and an acquisition function, then runs complete optimisation campaigns and compares them with random search.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, RBF, WhiteKernel, ConstantKernel as C
rng = np.random.default_rng(3)
```

## Gaussian processes

A Gaussian process (GP) {cite:p}`rasmussen2006` predicts a mean **and** an uncertainty at every point. Instead of assuming a formula such as a straight line or a parabola, a GP assumes only that the response is *smooth*: points close together have similar values. That assumption is encoded in a **kernel**, a function that says how strongly two inputs are correlated as a function of the distance between them; its **length scale** sets how far apart two points can be and still influence each other. Near a measurement the GP's prediction is pinned close to the measured value and the uncertainty is small; far from all measurements the prediction relaxes towards the average and the uncertainty grows. scikit-learn's `GaussianProcessRegressor` fits the kernel's settings to the data automatically. A `WhiteKernel` term represents measurement noise, and `normalize_y=True` rescales the target internally.

A two-point example shows the key behaviour, small uncertainty at the data and large uncertainty between them:

```{code-cell} ipython3
gp_demo = GaussianProcessRegressor(RBF(1.0), optimizer=None).fit([[0.0], [4.0]], [1.0, 2.0])
for x_q in [0.0, 2.0, 4.0]:
    m_q, s_q = gp_demo.predict([[x_q]], return_std=True)
    print(f"x = {x_q}: mean {m_q[0]:.2f}, uncertainty {s_q[0]:.2f}")
```

Here a hidden function plays the role of a real experiment: yield as a function of temperature, with a large peak near 92 °C and a smaller local maximum near 40 °C. After four experiments, the GP has a rough idea of the shape and knows where it is unsure.

```{code-cell} ipython3
def true_yield(T):           # a hidden "experiment": yield vs temperature (°C)
    return 80 * np.exp(-0.5 * ((T - 92) / 18) ** 2) + 15 * np.exp(-0.5 * ((T - 40) / 8) ** 2)

T_obs = np.array([30, 60, 120, 140.0])
y_obs = true_yield(T_obs) + rng.normal(0, 1.5, T_obs.size)
grid = np.linspace(20, 150, 400)

gp = GaussianProcessRegressor(C(1000) * Matern(length_scale=20, nu=2.5) + WhiteKernel(2), normalize_y=True)
gp.fit(T_obs[:, None], y_obs)
mu, sd = gp.predict(grid[:, None], return_std=True)

plt.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color="#6c4ad6", alpha=0.2, label="95 % band")
plt.plot(grid, mu, color="#6c4ad6", label="GP mean")
plt.plot(grid, true_yield(grid), "k:", label="truth (unknown)")
plt.plot(T_obs, y_obs, "o", color="#e5487a", ms=9, label="experiments")
plt.xlabel("T / °C"); plt.ylabel("yield / %"); plt.legend(); plt.title("A GP after four experiments"); plt.show()
```

## Acquisition functions

The GP says what might happen at each condition; the **acquisition function** turns that into a decision about which experiment to run next. It gives each candidate a score that is high where the predicted mean is good (exploitation) or where the uncertainty is large (exploration), and the next experiment is the candidate with the highest score. **Expected improvement** (EI) asks: by how much do we expect to beat the current best? It averages the possible improvement over the GP's predicted distribution at each point, so a point with a slightly lower mean but a much larger uncertainty can still win.

$$\mathrm{EI}(x) = (\mu - f^* - \xi)\,\Phi(z) + \sigma\,\phi(z), \qquad z = \frac{\mu - f^* - \xi}{\sigma}$$

Here $f^*$ is the best result so far, $\Phi$ and $\phi$ are the cumulative and density functions of the normal distribution, and the small $\xi$ adds a little extra exploration. **Upper confidence bound** (UCB) is simpler: it adds κ standard deviations to the mean, so larger κ means more exploration.

```{code-cell} ipython3
def expected_improvement(mu, sd, best, xi=0.01):
    sd = np.maximum(sd, 1e-9)
    z = (mu - best - xi) / sd
    return (mu - best - xi) * norm.cdf(z) + sd * norm.pdf(z)

def ucb(mu, sd, kappa=2.0):
    return mu + kappa * sd
```

## The BO loop

Bayesian optimisation repeats four steps: fit the GP to all results so far, compute the acquisition function over the candidates, run the experiment at its maximum, and add the result to the data. Each panel below shows one iteration: the GP (top) and EI (bottom), with the dashed line marking the next experiment. Watch how the first choices explore regions of high uncertainty and later ones home in on the peak.

```{code-cell} ipython3
T_run, y_run = list(T_obs), list(y_obs)
fig, axes = plt.subplots(2, 4, figsize=(16, 6), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
for it in range(4):
    gp.fit(np.array(T_run)[:, None], np.array(y_run))
    mu, sd = gp.predict(grid[:, None], return_std=True)
    ei = expected_improvement(mu, sd, max(y_run))
    nxt = grid[np.argmax(ei)]
    ax, ax2 = axes[0, it], axes[1, it]
    ax.fill_between(grid, mu - 2 * sd, mu + 2 * sd, color="#6c4ad6", alpha=0.2)
    ax.plot(grid, mu, color="#6c4ad6"); ax.plot(grid, true_yield(grid), "k:", lw=1)
    ax.plot(T_run, y_run, "o", color="#e5487a"); ax.axvline(nxt, color="#0f9d8a", ls="--")
    ax.set_title(f"iteration {it + 1}: best so far {max(y_run):.1f} %")
    ax2.fill_between(grid, ei, color="#0f9d8a", alpha=0.5); ax2.set_xlabel("T / °C")
    T_run.append(nxt); y_run.append(true_yield(nxt) + rng.normal(0, 1.5))
axes[0, 0].set_ylabel("yield / %"); axes[1, 0].set_ylabel("EI")
plt.tight_layout(); plt.show()
```

## Two dimensions: temperature and catalyst loading

Real optimisations have several variables, and their interactions are exactly what one-variable-at-a-time searches miss. The simulated reaction below depends on temperature and catalyst loading, with a broad main optimum and a narrow secondary one. Two practical details matter. The variables have very different ranges (tens of degrees versus a few mol %), so they are scaled to 0–1 before fitting, and each variable gets its own length scale. And instead of optimising EI continuously, it is evaluated on 3000 random candidate points and the best is chosen, a simple and robust approach. Each campaign starts with five random experiments, then lets BO choose twenty more.

```{code-cell} ipython3
def experiment(T, load):
    return (75 * np.exp(-((T - 85) / 25) ** 2 - ((load - 2.5) / 1.2) ** 2) + 20 * np.exp(-((T - 130) / 10) ** 2 - ((load - 0.6) / 0.4) ** 2)
            + rng.normal(0, 1.0, np.shape(T)))

lo, hi = np.array([40, 0.1]), np.array([150, 5.0])
def scale(X): return (X - lo) / (hi - lo)
cand = lo + (hi - lo) * rng.random((3000, 2))

def run_campaign(strategy, n_init=5, n_iter=20, seed=0):
    r = np.random.default_rng(seed)
    X = lo + (hi - lo) * r.random((n_init, 2)); y = experiment(X[:, 0], X[:, 1])
    for _ in range(n_iter):
        if strategy == "random":
            x = lo + (hi - lo) * r.random(2)
        else:
            g = GaussianProcessRegressor(C(1) * Matern(length_scale=[0.3, 0.3], nu=2.5) + WhiteKernel(1e-2), normalize_y=True)
            g.fit(scale(X), y)
            m, s = g.predict(scale(cand), return_std=True)
            x = cand[np.argmax(expected_improvement(m, s, y.max()))]
        X = np.vstack([X, x]); y = np.append(y, experiment(*x))
    return X, y

X_bo, y_bo = run_campaign("bo")
TT, LL = np.meshgrid(np.linspace(40, 150, 150), np.linspace(0.1, 5, 150))
Z = 75 * np.exp(-((TT - 85) / 25) ** 2 - ((LL - 2.5) / 1.2) ** 2) + 20 * np.exp(-((TT - 130) / 10) ** 2 - ((LL - 0.6) / 0.4) ** 2)
plt.figure(figsize=(7, 5))
plt.contourf(TT, LL, Z, levels=20, cmap="viridis"); plt.colorbar(label="true yield / %")
plt.scatter(X_bo[:5, 0], X_bo[:5, 1], c="white", edgecolor="k", s=60, label="initial random")
plt.scatter(X_bo[5:, 0], X_bo[5:, 1], c=np.arange(len(X_bo) - 5), cmap="autumn", edgecolor="k", s=70, label="BO choices")
plt.xlabel("T / °C"); plt.ylabel("catalyst loading / mol %"); plt.legend(loc="upper right"); plt.title("Where BO chose to experiment")
plt.show()
```

### Is it better than random?

A single campaign could be lucky, so a fair comparison repeats it with several random seeds and plots the best result found so far against the number of experiments, with the range across seeds shaded.

```{code-cell} ipython3
curves = {"random": [], "bo": []}
for seed in range(6):
    for s in curves:
        _, y = run_campaign(s, seed=seed)
        curves[s].append(np.maximum.accumulate(y))
for s, c in [("random", "#e5487a"), ("bo", "#0f9d8a")]:
    arr = np.array(curves[s])
    plt.plot(arr.mean(0), color=c, lw=2.5, label="Bayesian optimisation" if s == "bo" else "random search")
    plt.fill_between(range(arr.shape[1]), arr.min(0), arr.max(0), color=c, alpha=0.15)
plt.xlabel("number of experiments"); plt.ylabel("best yield found / %"); plt.legend(); plt.title("BO finds the optimum in fewer runs")
plt.show()
```

## Categorical variables: choosing ligand, base and additive

Many choices in chemistry are categories rather than numbers: which ligand, which base, which solvent. A GP needs numbers, so categories are turned into **one-hot** columns (Chapter 32) with `pd.get_dummies`, optionally supplemented by physical descriptors of each choice, such as a ligand's cone angle or a base's pKa, which tell the model that some categories are more alike than others. Real reaction optimisation mixes continuous and categorical choices. Here we optimise over the simulated C–N coupling screen in `cn_coupling_hte_sim.csv`, pretending each of its 960 conditions costs one experiment. Similar campaigns on real high-throughput data were reported by Shields and co-workers {cite:p}`shields2021`.

```{code-cell} ipython3
DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name)

hte = load_data("cn_coupling_hte_sim.csv")
Xh = pd.get_dummies(hte[["aryl_halide", "ligand", "base", "additive"]]).values.astype(float)
Xh = np.hstack([Xh, hte[["ligand_electron_richness", "ligand_cone_angle", "base_pKa"]].values / [1, 200, 20]])
yh = hte.yield_sim.values

def hte_campaign(strategy, budget=40, seed=0):
    r = np.random.default_rng(seed)
    done = list(r.choice(len(yh), 8, replace=False))
    while len(done) < budget:
        if strategy == "random":
            nxt = r.choice(list(set(range(len(yh))) - set(done)))
        else:
            g = GaussianProcessRegressor(C(1) * RBF(3.0) + WhiteKernel(0.05), normalize_y=True).fit(Xh[done], yh[done])
            m, s = g.predict(Xh, return_std=True)
            ei = expected_improvement(m, s, yh[done].max()); ei[done] = -1
            nxt = int(np.argmax(ei))
        done.append(nxt)
    return np.maximum.accumulate(yh[done])

for s, c in [("random", "#e5487a"), ("bo", "#0f9d8a")]:
    arr = np.array([hte_campaign(s, seed=k) for k in range(5)])
    plt.plot(arr.mean(0), color=c, lw=2.5, label=s)
plt.axhline(yh.max(), ls=":", color="k", label="best in the full screen")
plt.xlabel("experiments run"); plt.ylabel("best yield / %"); plt.legend(); plt.title("Optimising a 960-condition screen"); plt.show()
```

## Active learning for models

The same idea builds better **models** with fewer labels: query the points where the model is most uncertain. In **active learning** the goal is not the single best condition but an accurate model everywhere, for example a property predictor for a whole chemical space, when each label (a measurement or a quantum calculation) is expensive. The simplest strategy is *uncertainty sampling*: always label the point where the model's uncertainty is largest. Below it is compared with random labelling on a wavy one-dimensional function.

```{code-cell} ipython3
x_pool = np.linspace(0, 10, 500)[:, None]
f = lambda x: np.sin(x) * x / 3
labelled = [10, 480]
errs = {"uncertainty": [], "random": []}
for strat in errs:
    lab = list(labelled); r = np.random.default_rng(0)
    for _ in range(15):
        g = GaussianProcessRegressor(RBF(1.0) + WhiteKernel(1e-3)).fit(x_pool[lab], f(x_pool[lab]).ravel())
        m, s = g.predict(x_pool, return_std=True)
        errs[strat].append(np.abs(m - f(x_pool).ravel()).mean())
        lab.append(int(np.argmax(s)) if strat == "uncertainty" else int(r.integers(len(x_pool))))
plt.semilogy(errs["uncertainty"], "o-", color="#0f9d8a", label="query most uncertain")
plt.semilogy(errs["random"], "s-", color="#e5487a", label="query at random")
plt.xlabel("labels acquired"); plt.ylabel("model error"); plt.legend(); plt.title("Active learning"); plt.show()
```

## Design of experiments

Before any model exists, space-filling designs give the best first batch. **Design of experiments** (DoE) is the statistical discipline of choosing experimental conditions to learn as much as possible from as few runs as possible. Purely random points tend to clump together and leave gaps. A **Latin hypercube** divides each variable's range into as many slices as there are points and puts exactly one point in each slice of every variable, so no value is repeated. **Sobol** sequences are *quasi-random*: deterministic sequences designed to fill space evenly (a power of two points works best). `scipy.stats.qmc` generates both.

```{code-cell} ipython3
from scipy.stats import qmc
designs = {"random": rng.random((20, 2)), "Latin hypercube": qmc.LatinHypercube(d=2, seed=0).random(20),
           "Sobol": qmc.Sobol(d=2, seed=0).random(16)}
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, (name, pts), c in zip(axes, designs.items(), ["#e5487a", "#6c4ad6", "#0f9d8a"]):
    ax.scatter(*pts.T, color=c, s=50, edgecolor="k"); ax.set(title=name, xlim=(0, 1), ylim=(0, 1)); ax.set_aspect("equal")
plt.tight_layout(); plt.show()
```

```{admonition} Summary
:class: tip
- Bayesian optimisation alternates between a probabilistic model and an acquisition function to choose each experiment.
- A Gaussian process predicts a mean and an uncertainty; its kernel and length scale encode smoothness.
- Expected improvement and UCB balance exploitation (good mean) and exploration (high uncertainty).
- Scale continuous variables; one-hot encode categories, adding descriptors where possible.
- Compare with random search over several seeds.
- Active learning labels the most uncertain points to build accurate models cheaply.
- Latin hypercube and Sobol designs give space-filling first batches.
```

## Exercises

**★ 1.** Replace EI with UCB (κ = 0.5, 2, 5). How does κ change the balance between exploration and exploitation?

**★★ 2.** Add a third variable, reaction time, to the 2D campaign and compare BO with a Latin-hypercube batch of the same size.

**★★★ 3.** Implement batch BO (choose four experiments per round) using the "kriging believer" heuristic.
