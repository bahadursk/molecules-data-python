---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# SymPy: Symbolic Chemistry

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Part I. Basic calculus helps.
```

NumPy works with numbers; **SymPy** works with **symbols**. It rearranges equations, differentiates, integrates and solves exactly, then prints the result as beautiful mathematics.

```{code-cell} ipython3
import sympy as sp
sp.init_printing()
```

## Symbols and expressions

```{code-cell} ipython3
P, V, n, R, T = sp.symbols("P V n R T", positive=True)
ideal = sp.Eq(P * V, n * R * T)
ideal
```

```{code-cell} ipython3
sp.solve(ideal, V)[0]
```

Substitute numbers:

```{code-cell} ipython3
V_expr = sp.solve(ideal, V)[0]
V_expr.subs({n: 1, R: 8.314, T: 273.15, P: 101325}).evalf(5)
```

## Solving equilibria exactly: ICE tables

For a weak acid HA with initial concentration $C$:

```{code-cell} ipython3
x, C, Ka = sp.symbols("x C K_a", positive=True)
eq = sp.Eq(x**2 / (C - x), Ka)
sols = sp.solve(eq, x)
sols
```

```{code-cell} ipython3
H = sols[0]
pH = -sp.log(H.subs({C: 0.1, Ka: 1.8e-5}), 10)
print("pH =", sp.N(pH, 4))
```

The approximation $x \approx \sqrt{K_a C}$ comes from a series expansion for small $K_a$:

```{code-cell} ipython3
sp.series(H, Ka, 0, 2)
```

## Calculus with rate laws

The integrated second-order rate law from the differential form $-\mathrm{d}[A]/\mathrm{d}t = k[A]^2$:

```{code-cell} ipython3
t, k, A0 = sp.symbols("t k A_0", positive=True)
A = sp.Function("A")
ode = sp.Eq(A(t).diff(t), -k * A(t)**2)
sol = sp.dsolve(ode, ics={A(0): A0})
sol
```

Half-life:

```{code-cell} ipython3
t_half = sp.solve(sp.Eq(sol.rhs, A0 / 2), t)[0]
t_half
```

Consecutive reactions A → B → C:

```{code-cell} ipython3
k1, k2 = sp.symbols("k_1 k_2", positive=True)
a, b = sp.Function("a"), sp.Function("b")
system = [sp.Eq(a(t).diff(t), -k1 * a(t)), sp.Eq(b(t).diff(t), k1 * a(t) - k2 * b(t))]
s = sp.dsolve(system, ics={a(0): 1, b(0): 0})
B_t = sp.simplify(s[1].rhs)
B_t
```

When is B at its maximum?

```{code-cell} ipython3
t_max = sp.solve(sp.diff(B_t, t), t)
sp.simplify(t_max[0])
```

```{code-cell} ipython3
import numpy as np, matplotlib.pyplot as plt
f = sp.lambdify(t, B_t.subs({k1: 1.0, k2: 0.25}), "numpy")
tt = np.linspace(0, 15, 300)
plt.plot(tt, np.exp(-tt), color="#e5487a", label="A")
plt.plot(tt, f(tt), color="#f2a93b", label="B")
plt.plot(tt, 1 - np.exp(-tt) - f(tt), color="#0f9d8a", label="C")
tm = float(t_max[0].subs({k1: 1.0, k2: 0.25}))
plt.axvline(tm, ls=":", color="k"); plt.text(tm + 0.2, 0.9, f"t_max = {tm:.2f}")
plt.xlabel("t"); plt.ylabel("fraction"); plt.legend(); plt.title("A → B → C, solved symbolically")
plt.show()
```

`lambdify` turns a symbolic expression into a fast NumPy function: derive with SymPy, compute with NumPy.

## Thermodynamics: Clausius–Clapeyron

```{code-cell} ipython3
P1, P2, T1, T2, dH = sp.symbols("P_1 P_2 T_1 T_2 \\Delta{}H_{vap}", positive=True)
cc = sp.Eq(sp.log(P2 / P1), -dH / R * (1 / T2 - 1 / T1))
sol_dH = sp.solve(cc, dH)[0]
display(sol_dH)
print("ΔHvap(water) ≈", sp.N(sol_dH.subs({P1: 3.17, P2: 101.3, T1: 298.15, T2: 373.15, R: 8.314}) / 1000, 4), "kJ/mol")
```

## Quantum mechanics: particle in a box

```{code-cell} ipython3
xq, L = sp.symbols("x L", positive=True)
nq = sp.symbols("n", integer=True, positive=True)
psi = sp.sqrt(2 / L) * sp.sin(nq * sp.pi * xq / L)
print("normalised:", sp.simplify(sp.integrate(psi**2, (xq, 0, L))))
hbar, m = sp.symbols("hbar m", positive=True)
E = sp.simplify(-hbar**2 / (2 * m) * sp.diff(psi, xq, 2) / psi)
E
```

```{code-cell} ipython3
sp.simplify(sp.integrate(psi * xq**2 * psi, (xq, 0, L)) - sp.integrate(psi * xq * psi, (xq, 0, L))**2)
```

That is the variance of position, $\langle x^2\rangle - \langle x\rangle^2$, in exact form.

## Balancing equations with exact matrices

```{code-cell} ipython3
#               KMnO4  HCl  KCl  MnCl2  H2O  Cl2
M = sp.Matrix([[ 1,    0,  -1,   0,    0,   0],   # K
               [ 1,    0,   0,  -1,    0,   0],   # Mn
               [ 4,    0,   0,   0,   -1,   0],   # O
               [ 0,    1,   0,   0,   -2,   0],   # H
               [ 0,    1,  -1,  -2,    0,  -2]])  # Cl
ns = M.nullspace()[0]
coeffs = ns / sp.gcd(list(ns)) * sp.ilcm(*[term.q for term in ns])
print(list(coeffs))   # 2 KMnO4 + 16 HCl -> 2 KCl + 2 MnCl2 + 8 H2O + 5 Cl2
```

## LaTeX output

```{code-cell} ipython3
print(sp.latex(sol.rhs))
```

Paste the output directly into a report or thesis.

## Exercises

**★ 1.** Solve the van der Waals equation $(P + a n^2/V^2)(V - nb) = nRT$ for $P$.

**★★ 2.** Derive the integrated rate law for a zero-order reaction and its half-life.

**★★★ 3.** Find the most probable radius for a hydrogen 2p electron by differentiating $r^2 R_{21}(r)^2$ with $R_{21} \propto r e^{-r/2a_0}$.
