---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Chemical Kinetics and ODEs

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 12, 13 and 16.
```

Rate laws are **differential equations**. Only the simplest have neat textbook solutions; everything else, from enzyme cascades to oscillating reactions, is solved numerically. SciPy's `solve_ivp` makes this routine.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.optimize import curve_fit
COL = ["#e5487a", "#f2a93b", "#0f9d8a", "#6c4ad6", "#3b82f6"]
```

## Integrated rate laws

```{code-cell} ipython3
t = np.linspace(0, 10, 200)
A0, k = 1.0, 0.5
orders = {"zero order": np.clip(A0 - k * 0.2 * t, 0, None), "first order": A0 * np.exp(-k * t),
          "second order": A0 / (1 + k * A0 * t)}
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
for (name, A), c in zip(orders.items(), COL):
    axes[0].plot(t, A, color=c, label=name)
axes[0].set(xlabel="t", ylabel="[A]", title="Concentration"); axes[0].legend()
axes[1].plot(t, np.log(orders["first order"]), color=COL[1]); axes[1].set(title="ln[A] linear: 1st order", xlabel="t")
axes[2].plot(t, 1 / orders["second order"], color=COL[2]); axes[2].set(title="1/[A] linear: 2nd order", xlabel="t")
plt.tight_layout(); plt.show()
```

## Numerical solution with `solve_ivp`

For A → B → C with rate constants $k_1, k_2$:

$$\frac{d[A]}{dt} = -k_1[A], \quad \frac{d[B]}{dt} = k_1[A] - k_2[B], \quad \frac{d[C]}{dt} = k_2[B]$$

```{code-cell} ipython3
def consecutive(t, y, k1, k2):
    A, B, C = y
    return [-k1 * A, k1 * A - k2 * B, k2 * B]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, (k1, k2) in zip(axes, [(1, 0.1), (1, 1.0), (0.1, 1)]):
    sol = solve_ivp(consecutive, (0, 30), [1, 0, 0], args=(k1, k2), dense_output=True)
    tt = np.linspace(0, 30, 300)
    for y, lab, c in zip(sol.sol(tt), "ABC", COL):
        ax.plot(tt, y, color=c, lw=2, label=lab)
    ax.set(title=f"k₁={k1}, k₂={k2}", xlabel="t")
axes[0].set_ylabel("concentration"); axes[0].legend(); plt.tight_layout(); plt.show()
```

When $k_2 \gg k_1$ (right), B never builds up: this is the regime where the **steady-state approximation** works.

## Parallel reactions and selectivity

```{code-cell} ipython3
def parallel(t, y, k1, k2):
    A, B, C = y
    return [-(k1 + k2) * A, k1 * A, k2 * A]

T = np.linspace(280, 380, 6)
Ea1, Ea2, Af = 50e3, 65e3, 1e8
fig, ax = plt.subplots(figsize=(6.5, 4))
for Ti, c in zip(T, plt.cm.coolwarm(np.linspace(0, 1, len(T)))):
    k1, k2 = Af * np.exp(-Ea1 / (8.314 * Ti)), Af * np.exp(-Ea2 / (8.314 * Ti))
    s = solve_ivp(parallel, (0, 1e4), [1, 0, 0], args=(k1, k2))
    ax.bar(f"{Ti:.0f} K", s.y[1, -1] / (s.y[1, -1] + s.y[2, -1]), color=c)
ax.set(ylabel="selectivity for B", title="Higher T favours the high-barrier product C")
plt.show()
```

## Pre-equilibrium and enzyme kinetics

E + S ⇌ ES → E + P, solved in full and compared with the Michaelis–Menten approximation:

```{code-cell} ipython3
def enzyme(t, y, k1, km1, k2):
    E, S, ES, P = y
    r1, rm1, r2 = k1 * E * S, km1 * ES, k2 * ES
    return [-r1 + rm1 + r2, -r1 + rm1, r1 - rm1 - r2, r2]

k1, km1, k2, E0, S0 = 1e6, 1e3, 1e2, 1e-6, 1e-3
sol = solve_ivp(enzyme, (0, 30), [E0, S0, 0, 0], args=(k1, km1, k2), method="LSODA", dense_output=True)
tt = np.linspace(0, 30, 400)
E, S, ES, P = sol.sol(tt)
KM, Vmax = (km1 + k2) / k1, k2 * E0
fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
ax[0].plot(tt, S * 1e3, color=COL[0], label="S"); ax[0].plot(tt, P * 1e3, color=COL[2], label="P")
ax[0].set(xlabel="t / s", ylabel="mM", title="Full mechanism"); ax[0].legend()
ax[1].plot(tt, ES / E0, color=COL[3]); ax[1].set(xlabel="t / s", ylabel="[ES]/[E]₀", title="Quasi-steady state of ES")
plt.tight_layout(); plt.show()
print(f"KM = {KM*1e3:.2f} mM, Vmax = {Vmax*1e6:.0f} μM/s")
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Systems with very different time scales are **stiff**. The default `RK45` solver becomes painfully slow on them; use `method="LSODA"`, `"BDF"` or `"Radau"`.
```

## Oscillating reactions: the Oregonator

The Belousov–Zhabotinsky reaction oscillates between red and blue for minutes. The scaled **Oregonator** model captures it with three variables.

```{code-cell} ipython3
def oregonator(t, y, eps=0.04, q=8e-4, f=1.0, epsp=2e-4):
    x, yy, z = y
    return [(q * yy - x * yy + x * (1 - x)) / eps, (-q * yy - x * yy + f * z) / epsp, x - z]

sol = solve_ivp(oregonator, (0, 30), [0.1, 0.1, 0.1], method="Radau", rtol=1e-8, atol=1e-10, dense_output=True)
tt = np.linspace(5, 30, 4000)
x, y, z = sol.sol(tt)
fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
ax[0].semilogy(tt, x, color=COL[0], label="HBrO₂ (x)"); ax[0].semilogy(tt, z, color=COL[4], label="Ce(IV) (z)")
ax[0].legend(); ax[0].set_ylabel("scaled conc.")
ax[1].imshow(z[None, :], aspect="auto", cmap="RdBu_r", extent=[tt[0], tt[-1], 0, 1]); ax[1].set_yticks([])
ax[1].set_xlabel("scaled time"); ax[1].set_title("Colour of the solution: ferroin red ↔ blue", fontsize=10)
plt.tight_layout(); plt.show()
```

## Fitting kinetic data

Fit rate constants of A → B → C to noisy measurements of B by wrapping `solve_ivp` inside `curve_fit`.

```{code-cell} ipython3
rng = np.random.default_rng(0)
t_obs = np.linspace(0, 12, 25)
true = (0.8, 0.25)
B_true = solve_ivp(consecutive, (0, 12), [1, 0, 0], args=true, t_eval=t_obs).y[1]
B_obs = B_true + rng.normal(0, 0.015, t_obs.size)

def model_B(t, k1, k2):
    return solve_ivp(consecutive, (0, t.max()), [1, 0, 0], args=(k1, k2), t_eval=t).y[1]

popt, pcov = curve_fit(model_B, t_obs, B_obs, p0=[0.5, 0.5], bounds=(0, 10))
err = np.sqrt(np.diag(pcov))
tt = np.linspace(0, 12, 200)
plt.plot(t_obs, B_obs, "o", color=COL[1], label="data")
plt.plot(tt, model_B(tt, *popt), color=COL[3], label=f"fit k₁={popt[0]:.2f}±{err[0]:.2f}, k₂={popt[1]:.2f}±{err[1]:.2f}")
plt.xlabel("t / min"); plt.ylabel("[B]"); plt.legend(); plt.title("Fitting a mechanism to data"); plt.show()
```

## Stochastic kinetics: the Gillespie algorithm

With only a few molecules (inside a cell, on a catalyst particle), concentrations fluctuate. The Gillespie algorithm simulates individual reaction events.

```{code-cell} ipython3
def gillespie(n0, k, t_end, rng):
    t, n, ts, ns = 0.0, n0, [0.0], [n0]
    while t < t_end and n > 0:
        rate = k * n
        t += rng.exponential(1 / rate)
        n -= 1
        ts.append(t); ns.append(n)
    return ts, ns

fig, ax = plt.subplots(figsize=(7, 4))
for i in range(8):
    ts, ns = gillespie(30, 0.3, 20, np.random.default_rng(i))
    ax.step(ts, ns, where="post", alpha=0.7)
tt = np.linspace(0, 20, 200)
ax.plot(tt, 30 * np.exp(-0.3 * tt), "k--", lw=2, label="deterministic")
ax.set(xlabel="t", ylabel="molecules of A", title="Stochastic first-order decay of 30 molecules"); ax.legend()
plt.show()
```

## Exercises

**★ 1.** Plot [A] for a first-order reaction with half-lives of 1, 5 and 10 min on one graph.

**★★ 2.** Model the Lotka–Volterra "autocatalytic" scheme A + X → 2X, X + Y → 2Y, Y → P and plot X against Y.

**★★★ 3.** Fit k₁, k₋₁ and k₂ of the enzyme mechanism to simulated [P]$(t)$ data. Which parameters are poorly determined, and why?

```{admonition} 🤖 ML corner
:class: ml-corner
In Chapter 50 we combine mechanistic ODE models like these with neural networks ("hybrid" or "grey-box" models) and build a digital twin of a reactor.
```
