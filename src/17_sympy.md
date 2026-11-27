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

NumPy and SciPy work with numbers: give them values and they give you a numerical answer. **SymPy** works with **symbols**, letters that stand for quantities, just as you do with pen and paper. It can rearrange the ideal gas law for $V$, differentiate a rate law, integrate a wavefunction or solve an equilibrium expression *exactly*, and then print the result as properly typeset mathematics. This is called **computer algebra**. It is useful whenever you want a general formula rather than one number, when you want to check a derivation from a textbook, or when you want to derive an expression once and then evaluate it quickly for many values. This chapter shows how to create symbols and equations, solve them, do calculus with rate laws and quantum mechanics, and pass the results on to NumPy.

SymPy is imported as `sp`, and `init_printing()` makes Jupyter display results as mathematics:

```{code-cell} ipython3
import sympy as sp
sp.init_printing()
```

## Symbols and expressions

In ordinary Python, writing `x + 1` fails unless `x` already has a value. In SymPy you first declare `x` as a **symbol** with `sp.symbols`, and then expressions containing it stay unevaluated, as algebra. Telling SymPy what you know about a symbol, for example `positive=True` for a pressure or a temperature, lets it simplify further, because it can then rule out negative or complex roots.

The most basic expression behaves just like algebra on paper:

```{code-cell} ipython3
x = sp.symbols("x")
expr = (x + 1)**2
print(expr)
print(sp.expand(expr))            # multiply out
print(sp.factor(x**2 - 1))        # and factorise
```

An equation is written with `sp.Eq(left, right)`, because `=` already means assignment and `==` tests whether two expressions are identical. Here is the ideal gas law:

```{code-cell} ipython3
P, V, n, R, T = sp.symbols("P V n R T", positive=True)
ideal = sp.Eq(P * V, n * R * T)
ideal
```

`sp.solve(equation, symbol)` rearranges an equation for the symbol you choose and returns a list of solutions; here there is only one.

```{code-cell} ipython3
sp.solve(ideal, V)[0]
```

To get a number, substitute values for the symbols with `.subs({symbol: value})`, and convert the exact result to a decimal with `.evalf(digits)`. One mole at 0 °C and 1 atm gives the familiar molar volume in m³:

```{code-cell} ipython3
V_expr = sp.solve(ideal, V)[0]
V_expr.subs({n: 1, R: 8.314, T: 273.15, P: 101325}).evalf(5)
```

## Solving equilibria exactly: ICE tables

An **ICE table** (Initial, Change, Equilibrium) is the standard way to set up an equilibrium problem: if $x$ mol L⁻¹ of a weak acid HA dissociates, the equilibrium concentrations are $[\mathrm{H^+}] = [\mathrm{A^-}] = x$ and $[\mathrm{HA}] = C - x$, so $K_a = x^2/(C - x)$. Chapter 6 solved this quadratic numerically. SymPy solves it in general, giving a formula valid for any $C$ and $K_a$; because $x$ was declared positive, only the physical root is returned.

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

Textbooks often use the approximation $x \approx \sqrt{K_a C}$. Where does it come from? A **series expansion** writes an expression as a sum of powers of a small quantity, here $K_a$. The leading term of the expansion is exactly $\sqrt{K_a C}$, and the next term shows the size of the correction, so SymPy both derives the approximation and tells you when it is safe:

```{code-cell} ipython3
sp.series(H, Ka, 0, 2)
```

## Calculus with rate laws

SymPy differentiates with `sp.diff(expr, x)` and integrates with `sp.integrate(expr, x)`, returning exact results. A first taste, with a first-order decay:

```{code-cell} ipython3
t, k, A0 = sp.symbols("t k A_0", positive=True)
conc = A0 * sp.exp(-k * t)
sp.diff(conc, t)                   # the rate of change, -k[A]
```

The derivative is $-k$ times the concentration, which is the first-order rate law: differentiating the integrated law gives back the differential law. SymPy can also go the other way and solve a **differential equation**, an equation relating a function to its derivatives. You declare an unknown function with `sp.Function`, write the equation with `.diff(t)`, and call `sp.dsolve` with the initial condition in `ics`. The integrated second-order rate law from the differential form $-\mathrm{d}[A]/\mathrm{d}t = k[A]^2$:

```{code-cell} ipython3
t, k, A0 = sp.symbols("t k A_0", positive=True)
A = sp.Function("A")
ode = sp.Eq(A(t).diff(t), -k * A(t)**2)
sol = sp.dsolve(ode, ics={A(0): A0})
sol
```

The **half-life** is the time at which the concentration has fallen to half its initial value. Setting the solution equal to $[A]_0/2$ and solving for $t$ gives the textbook result that a second-order half-life depends on the starting concentration, unlike a first-order one:

```{code-cell} ipython3
t_half = sp.solve(sp.Eq(sol.rhs, A0 / 2), t)[0]
t_half
```

Consecutive reactions A → B → C give a *system* of differential equations: A decays to B, and B is formed from A while decaying to C. `dsolve` accepts a list of equations and solves them together.

```{code-cell} ipython3
k1, k2 = sp.symbols("k_1 k_2", positive=True)
a, b = sp.Function("a"), sp.Function("b")
system = [sp.Eq(a(t).diff(t), -k1 * a(t)), sp.Eq(b(t).diff(t), k1 * a(t) - k2 * b(t))]
s = sp.dsolve(system, ics={a(0): 1, b(0): 0})
B_t = sp.simplify(s[1].rhs)
B_t
```

When is the intermediate B at its maximum? At a maximum the slope is zero, so we differentiate $[B](t)$ and solve for the time at which the derivative vanishes. The result, $t_\max = \ln(k_1/k_2)/(k_1 - k_2)$, is a classic of kinetics and tells a synthetic chemist exactly when to stop a reaction to collect the most intermediate.

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

The Clausius–Clapeyron equation links the vapour pressure of a liquid at two temperatures to its enthalpy of vaporisation. Solving it symbolically for $\Delta H_\mathrm{vap}$ gives a formula that can be used with any pair of measurements; substituting the vapour pressure of water at 25 °C (3.17 kPa) and at its boiling point (101.3 kPa) gives a value close to the tabulated 41 kJ mol⁻¹ near 100 °C. Symbol names can include LaTeX, as in `\Delta{}H_{vap}`, so the output looks like a textbook.

```{code-cell} ipython3
P1, P2, T1, T2, dH = sp.symbols("P_1 P_2 T_1 T_2 \\Delta{}H_{vap}", positive=True)
cc = sp.Eq(sp.log(P2 / P1), -dH / R * (1 / T2 - 1 / T1))
sol_dH = sp.solve(cc, dH)[0]
display(sol_dH)
print("ΔHvap(water) ≈", sp.N(sol_dH.subs({P1: 3.17, P2: 101.3, T1: 298.15, T2: 373.15, R: 8.314}) / 1000, 4), "kJ/mol")
```

## Quantum mechanics: particle in a box

The particle in a box is the simplest quantum system: a particle of mass $m$ confined to a line of length $L$. Its wavefunctions are $\psi_n = \sqrt{2/L}\sin(n\pi x/L)$. Two properties can be checked symbolically. A wavefunction must be **normalised**, meaning $\int_0^L \psi^2\,dx = 1$, so that the total probability of finding the particle is one. And applying the kinetic-energy operator $-\frac{\hbar^2}{2m}\frac{d^2}{dx^2}$ to $\psi$ must give a constant times $\psi$; that constant is the energy, $E_n = \frac{n^2\pi^2\hbar^2}{2mL^2}$. Declaring `n` as an integer lets SymPy simplify $\sin^2(n\pi) = 0$.

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

Balancing a chemical equation is a linear-algebra problem: each element gives one equation saying that atoms in equal atoms out, and the stoichiometric coefficients are the unknowns. Writing reactant counts as positive and product counts as negative, the coefficients form a vector that the matrix multiplies to zero, which is called the **null space** of the matrix. NumPy would give this as decimals; SymPy's exact `Matrix` gives exact fractions, which are then scaled to the smallest whole numbers. Here is the classic reaction of permanganate with hydrochloric acid:

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

**LaTeX** is the typesetting language used for mathematics in most scientific journals and theses. `sp.latex(expr)` converts any SymPy expression into LaTeX source, so a derived result can go straight into a document without retyping, and without the typos that retyping introduces.

```{code-cell} ipython3
print(sp.latex(sol.rhs))
```

Paste the output directly into a report or thesis.

```{admonition} Summary
:class: tip
- Declare symbols with `sp.symbols("x y", positive=True)`; write equations with `sp.Eq`.
- `sp.solve` rearranges and solves; `.subs()` substitutes values and `.evalf()` gives decimals.
- `sp.diff`, `sp.integrate` and `sp.series` do exact calculus; `sp.dsolve` solves differential equations such as rate laws.
- Exact `sp.Matrix` methods such as `nullspace` balance equations without rounding.
- `sp.lambdify` turns an expression into a fast NumPy function; `sp.latex` exports it for documents.
```

## Exercises

**★ 1.** Solve the van der Waals equation $(P + a n^2/V^2)(V - nb) = nRT$ for $P$.

**★★ 2.** Derive the integrated rate law for a zero-order reaction and its half-life.

**★★★ 3.** Find the most probable radius for a hydrogen 2p electron by differentiating $r^2 R_{21}(r)^2$ with $R_{21} \propto r e^{-r/2a_0}$.
