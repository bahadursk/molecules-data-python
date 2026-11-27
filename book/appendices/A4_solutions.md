---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Appendix D: Selected Solutions

Chapters 1–6 have their solutions in drop-down boxes directly under each exercise. This appendix gives worked solutions to a selection of exercises from later chapters, chosen because they exercise an idea that comes back again and again. Every cell below runs, so you can change the numbers and see what happens.

Try each exercise yourself before reading the answer. A solution you have struggled with first teaches far more than one you have only read.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
```

## Part I · Python

### Chapter 7, ★★ 2: the Balmer series as a generator

$1/\lambda = R_H(1/4 - 1/n^2)$ with $R_H = 1.0968 \times 10^7$ m⁻¹. The wavelengths fall towards the series limit of 364.6 nm, so "stop below 370 nm" terminates the generator.

```{code-cell} ipython3
R_H = 1.0968e7          # m^-1

def balmer(stop_nm=370.0):
    n = 3
    while True:
        lam_nm = 1e9 / (R_H * (1 / 4 - 1 / n ** 2))
        if lam_nm < stop_nm:
            return
        yield n, lam_nm
        n += 1

for n, lam in balmer():
    print(f"n = {n:2d} → 2   λ = {lam:6.1f} nm")
```

Hα (656 nm, red), Hβ (486 nm) and Hγ (434 nm) are the familiar visible lines.

### Chapter 9, ★★ 2: a `molar_mass` that fails clearly

```{code-cell} ipython3
import re

MASSES = {"H": 1.008, "C": 12.011, "N": 14.007, "O": 15.999, "S": 32.06, "Cl": 35.45, "Na": 22.990}

def molar_mass(formula):
    total = 0.0
    for sym, num in re.findall(r"([A-Z][a-z]?)(\d*)", formula):
        if sym not in MASSES:
            raise ValueError(f"unknown element symbol {sym!r} in {formula!r}")
        total += MASSES[sym] * (int(num) if num else 1)
    return total

print(f"{molar_mass('C6H12O6'):.3f} g/mol")
try:
    molar_mass("C6H12Xy")
except ValueError as e:
    print("ValueError:", e)
```

The error message names both the bad symbol and the whole formula, which is what someone debugging a 2 000-line CSV needs.

### Chapter 10, ★★ 2: degree of unsaturation

$DoU = C - H/2 + N/2 + 1$, with halogens counted as hydrogens.

```{code-cell} ipython3
def degree_of_unsaturation(counts):
    C = counts.get("C", 0)
    H = counts.get("H", 0) + sum(counts.get(x, 0) for x in ["F", "Cl", "Br", "I"])
    N = counts.get("N", 0)
    return C - H / 2 + N / 2 + 1

for name, counts in {"benzene C6H6": {"C": 6, "H": 6},
                     "cyclohexene C6H10": {"C": 6, "H": 10},
                     "pyridine C5H5N": {"C": 5, "H": 5, "N": 1},
                     "chlorobenzene C6H5Cl": {"C": 6, "H": 5, "Cl": 1}}.items():
    print(f"{name:22s} DoU = {degree_of_unsaturation(counts):.0f}")
```

Benzene's four degrees of unsaturation are one ring plus three C=C bonds.

## Part II · The scientific Python stack

### Chapter 12, ★★ 2: balancing Fe₂O₃ + C → Fe + CO₂ with the SVD

Write one row per element and one column per species, with products negative. The balanced coefficients are the null space of that matrix, which is the last right-singular vector.

```{code-cell} ipython3
#              Fe2O3   C   Fe  CO2
M = np.array([[2,      0, -1,  0],    # Fe
              [3,      0,  0, -2],    # O
              [0,      1,  0, -1]])   # C
_, s, Vt = np.linalg.svd(M)
v = Vt[-1]                                  # spans the null space (one reaction, so 1-D)
v = v / np.min(np.abs(v))                   # scale the smallest coefficient to 1
v = v * np.sign(v[0])                       # make the coefficients positive
coef = np.round(v * 2).astype(int)          # smallest whole numbers (here, multiply by 2)
coef = coef // np.gcd.reduce(coef)
print("coefficients:", coef)
print("check M @ coef =", M @ coef)
```

So $2\,\mathrm{Fe_2O_3} + 3\,\mathrm{C} \rightarrow 4\,\mathrm{Fe} + 3\,\mathrm{CO_2}$. The check row of zeros confirms every element balances.

### Chapter 12, ★★★ 3: Hückel theory of 1,3-butadiene

For a four-carbon chain, α on the diagonal and β between bonded neighbours. Working in units where α = 0 and β = −1, the eigenvalues are the orbital energies in units of |β|.

```{code-cell} ipython3
H = np.zeros((4, 4))
for i in range(3):
    H[i, i + 1] = H[i + 1, i] = -1.0
E, C = np.linalg.eigh(H)
print("orbital energies (units of |β|, α = 0):", np.round(E, 3))
homo, lumo = E[1], E[2]                     # 4 π electrons fill the two lowest orbitals
print(f"HOMO–LUMO gap = {lumo - homo:.3f} |β|")
```

The gap is $2 \times 0.618 = 1.236\,|\beta|$, where 0.618 is the golden ratio minus one, which is no coincidence: the eigenvalues of a chain are $2\cos(k\pi/(n+1))$.

### Chapter 16, ★ 1: exact pH of 0.010 M acetic acid

Solve the full charge balance $[\mathrm{H^+}] = [\mathrm{A^-}] + [\mathrm{OH^-}]$ with $[\mathrm{A^-}] = C K_a/(K_a + [\mathrm{H^+}])$.

```{code-cell} ipython3
from scipy import optimize

Ka, Kw, C = 10 ** -4.76, 1e-14, 0.010

def charge_balance(h):
    return h - C * Ka / (Ka + h) - Kw / h

h = optimize.brentq(charge_balance, 1e-12, 1.0)
approx = np.sqrt(Ka * C)
print(f"exact pH   = {-np.log10(h):.3f}")
print(f"√(Ka·C) pH = {-np.log10(approx):.3f}   (the usual approximation)")
```

The textbook approximation is good to about 0.01 pH units here because the acid is only about 4 % dissociated; at lower concentrations the two diverge.

### Chapter 16, ★★ 2: fitting a second-order rate law

```{code-cell} ipython3
from scipy.optimize import curve_fit

rng = np.random.default_rng(16)
k_true, A0 = 0.85, 0.050                         # L mol^-1 min^-1, mol L^-1
t = np.linspace(0, 60, 16)
A = A0 / (1 + k_true * A0 * t) * (1 + rng.normal(0, 0.015, t.size))

def second_order(t, k, A0):
    return A0 / (1 + k * A0 * t)

(k_fit, A0_fit), cov = curve_fit(second_order, t, A, p0=[1.0, 0.05])
k_err = np.sqrt(cov[0, 0])
print(f"k = {k_fit:.3f} ± {k_err:.3f} L mol⁻¹ min⁻¹   (true {k_true})")

fig, ax = plt.subplots(figsize=(5.5, 3.6))
ax.plot(t, 1 / A, "o", color="#6c4ad6", label="data")
ax.plot(t, 1 / second_order(t, k_fit, A0_fit), color="#0f9d8a", label="fit")
ax.set(xlabel="t / min", ylabel="1/[A]  /  L mol$^{-1}$", title="second order: 1/[A] is linear in t")
ax.legend(); plt.tight_layout(); plt.show()
```

Fit the integrated rate law directly, not the linearised $1/[A]$ plot: the linearisation distorts the noise and gives biased uncertainties.

### Chapter 17, ★★ 2 and ★★★ 3: zero-order kinetics and the 2p radius

```{code-cell} ipython3
import sympy as sp

t, k, A0 = sp.symbols("t k A_0", positive=True)
A = sp.Function("A")
sol = sp.dsolve(sp.Eq(A(t).diff(t), -k), A(t), ics={A(0): A0})
print("zero order:", sol)
t_half = sp.solve(sp.Eq(sol.rhs, A0 / 2), t)[0]
print("half-life  :", t_half)

r, a0 = sp.symbols("r a_0", positive=True)
P = r ** 2 * (r * sp.exp(-r / (2 * a0))) ** 2          # r² R21², up to a constant
r_max = sp.solve(sp.diff(P, r), r)
print("most probable radius of a 2p electron:", r_max)
```

The zero-order half-life $[A]_0/2k$ depends on the starting concentration, unlike first order. The most probable 2p radius is $4a_0$, matching the Bohr-model radius for $n = 2$.

### Chapter 18, ★★ 2 and ★★★ 3: propagating uncertainty

```{code-cell} ipython3
from uncertainties import ufloat, umath

V_naoh = ufloat(24.35, 0.05)        # mL
c_naoh = ufloat(0.1000, 0.0002)     # mol/L
V_hcl = ufloat(25.00, 0.03)         # mL
c_hcl = c_naoh * V_naoh / V_hcl
print(f"[HCl] = {c_hcl:.2uP} mol/L")
```

```{code-cell} ipython3
R, T = 8.314e-3, 298.15
dG = ufloat(-20.0, 2.0)
K_linear = umath.exp(-dG / (R * T))

rng = np.random.default_rng(18)
samples = np.exp(-rng.normal(-20.0, 2.0, 200_000) / (R * T))
lo, med, hi = np.percentile(samples, [16, 50, 84])
print(f"linear propagation: K = {K_linear.n:.0f} ± {K_linear.s:.0f}")
print(f"Monte Carlo       : median {med:.0f}, 68 % interval {lo:.0f} – {hi:.0f}")
```

Linear propagation assumes $K$ is roughly linear in ΔG over ±2 kJ/mol, but $e^{x}$ is strongly curved: a 2 kJ/mol shift changes $K$ by a factor of more than two. The Monte Carlo distribution is lopsided (log-normal), so a symmetric "±" is the wrong way to report it. Report $\log_{10} K$ with a symmetric error instead.

## Part III · Cheminformatics

### Chapter 19, ★★ 2: stereoisomers and their InChIKeys

```{code-cell} ipython3
from rdkit import Chem

for smi in ["N[C@@H](C)C(=O)O", "N[C@H](C)C(=O)O"]:
    m = Chem.MolFromSmiles(smi)
    print(f"{smi:20s} {Chem.MolToInchiKey(m)}")
```

The first 14 characters (the connectivity hash) are identical; the second block differs, because it encodes the stereochemistry layer. Searching by the first block alone finds both enantiomers of alanine, which is sometimes exactly what you want.

### Chapter 21, ★ 1 and ★★★ 3: a nitro SMARTS, and Tanimoto by hand

```{code-cell} ipython3
from rdkit import DataStructs
from rdkit.Chem import rdFingerprintGenerator

nitro = Chem.MolFromSmarts("[N+](=O)[O-]")
for name, smi in [("nitrobenzene", "c1ccccc1[N+](=O)[O-]"),
                  ("TNT", "Cc1c(cc(cc1[N+](=O)[O-])[N+](=O)[O-])[N+](=O)[O-]")]:
    print(f"{name:12s} nitro groups: {len(Chem.MolFromSmiles(smi).GetSubstructMatches(nitro))}")

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
a = gen.GetFingerprintAsNumPy(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")).astype(bool)   # aspirin
b = gen.GetFingerprintAsNumPy(Chem.MolFromSmiles("OC(=O)c1ccccc1O")).astype(bool)           # salicylic acid
mine = (a & b).sum() / (a | b).sum()
rdkit_val = DataStructs.TanimotoSimilarity(
    gen.GetFingerprint(Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")),
    gen.GetFingerprint(Chem.MolFromSmiles("OC(=O)c1ccccc1O")))
print(f"Tanimoto (NumPy) = {mine:.4f}   Tanimoto (RDKit) = {rdkit_val:.4f}")
```

Always write SMARTS for the *charge-separated* nitro form, because that is how RDKit stores it after sanitisation.

### Chapter 22, ★ 1: ester hydrolysis applied to aspirin

A first attempt, using `C` for the alkoxy carbon:

```{code-cell} ipython3
from rdkit.Chem import AllChem

def run(rxn_smarts, mol):
    rxn = AllChem.ReactionFromSmarts(rxn_smarts)
    out = set()
    for prods in rxn.RunReactants((mol,)):
        for p in prods:
            Chem.SanitizeMol(p)
        out.add(" + ".join(sorted(Chem.MolToSmiles(p) for p in prods)))
    return out

aspirin = Chem.MolFromSmiles("CC(=O)Oc1ccccc1C(=O)O")
print(run("[C:1](=[O:2])[O:3][C:4]>>[C:1](=[O:2])[OH].[OH:3][C:4]", aspirin))
```

Nothing happens. In SMARTS, upper-case `C` means *aliphatic* carbon, and aspirin's ester oxygen is bonded to an *aromatic* carbon (`c`). Use `[#6]`, "any carbon", instead:

```{code-cell} ipython3
print(run("[C:1](=[O:2])[O:3][#6:4]>>[C:1](=[O:2])[OH].[OH:3][#6:4]", aspirin))
```

One product set: acetic acid and salicylic acid, which is why old aspirin tablets smell of vinegar. The lesson generalises: when a reaction template "doesn't work", check aromatic versus aliphatic atoms first.

## Part IV · Chemistry applications

### Chapter 26, ★ 1 and ★★ 2: methane combustion and carbonic acid

Standard Gibbs energies of formation at 298 K (kJ mol⁻¹, widely tabulated values): CH₄(g) −50.5, CO₂(g) −394.4, H₂O(g) −228.6, O₂(g) 0.

```{code-cell} ipython3
dGf = {"CH4": -50.5, "O2": 0.0, "CO2": -394.4, "H2O(g)": -228.6}
dG = dGf["CO2"] + 2 * dGf["H2O(g)"] - dGf["CH4"] - 2 * dGf["O2"]
R, T = 8.314e-3, 298.15
log10K = -dG / (R * T * np.log(10))
print(f"ΔG° = {dG:.1f} kJ/mol,  log10 K = {log10K:.0f}")
```

$K \approx 10^{140}$ is too large for a float to be meaningful, which is why Chapter 2 recommends working with $\log_{10} K$.

```{code-cell} ipython3
pKa1, pKa2 = 6.35, 10.33
pH = np.linspace(2, 14, 1201)
h = 10 ** -pH
Ka1, Ka2 = 10 ** -pKa1, 10 ** -pKa2
D = h ** 2 + Ka1 * h + Ka1 * Ka2
alpha = np.vstack([h ** 2 / D, Ka1 * h / D, Ka1 * Ka2 / D])
best = pH[np.argmax(alpha[1])]
print(f"[HCO3-] is maximal at pH {best:.2f}; (pKa1 + pKa2)/2 = {(pKa1 + pKa2) / 2:.2f}")

fig, ax = plt.subplots(figsize=(6, 3.6))
for a, lbl, c in zip(alpha, ["H$_2$CO$_3$", "HCO$_3^-$", "CO$_3^{2-}$"], ["#6c4ad6", "#0f9d8a", "#f2a93b"]):
    ax.plot(pH, a, label=lbl, color=c, lw=2)
ax.axvline(best, ls=":", c="grey")
ax.set(xlabel="pH", ylabel="fraction", title="carbonic acid speciation"); ax.legend()
plt.tight_layout(); plt.show()
```

The amphiprotic species peaks at the average of the two pKa values, which is also the pH of a pure sodium bicarbonate solution (≈ 8.3).

### Chapter 28, ★★ 2: Hückel gaps of the acenes

```{code-cell} ipython3
from rdkit.Chem import rdmolops

acenes = {"benzene": "c1ccccc1", "naphthalene": "c1ccc2ccccc2c1",
          "anthracene": "c1ccc2cc3ccccc3cc2c1", "tetracene": "c1ccc2cc3cc4ccccc4cc3cc2c1"}
rows = []
for name, smi in acenes.items():
    A = rdmolops.GetAdjacencyMatrix(Chem.MolFromSmiles(smi)).astype(float)
    E = np.linalg.eigvalsh(-A)                 # α = 0, β = −1
    n_occ = A.shape[0] // 2
    rows.append(dict(molecule=name, n_C=A.shape[0], gap_beta=E[n_occ] - E[n_occ - 1]))
pd.DataFrame(rows).round(3)
```

The gap shrinks steadily as the conjugated system grows. Simple Hückel theory gets this trend right but not the absolute energies (it ignores electron repulsion, and |β| fitted to spectra differs from |β| fitted to thermochemistry). Experimentally, benzene, naphthalene and anthracene absorb in the ultraviolet and look colourless, while tetracene absorbs blue light and is orange: the shrinking gap pushes absorption into the visible, which is the trend this connectivity-matrix calculation predicts.

## Parts V–VII · Machine learning (written answers)

### Chapter 50, ★ 1: similar activation energies

If $E_{a,2}$ falls to 70 kJ/mol, close to $E_{a,1}$ = 65 kJ/mol, the ratio $k_2/k_1$ barely changes with temperature. The maximum attainable $[B]$ then depends very little on temperature, so the optimum curve flattens and the main effect of temperature is on *speed*, not *selectivity*. Selectivity is temperature-tunable only when the competing barriers differ.

### Chapter 50, ★ 2: random versus chronological splits

A random split mixes early and late batches, so the test set contains near-copies of training conditions and the reported R² is higher. Report the chronological number, because the soft sensor will always be used on *future* data, and write beside it the split used and the date range of the training data.

### Chapter 51, ★ 1: the metric is part of the explanation

Permutation importance measures how much a chosen score drops. ROC-AUC rewards correct *ranking*; accuracy rewards correct *classification at 0.5*. A feature that reorders borderline molecules can matter a lot for AUC and hardly at all for accuracy. So "which feature matters?" always means "which feature matters *for this metric*?", and the metric should be the one tied to your decision.

### Chapter 51, ★★ 2: choosing the refusal threshold

A tighter domain gate answers fewer molecules but gets more of them right (look at the risk–coverage curve). For a screening campaign, where a missed hit costs little, set the gate loose (answer 95–99 %). For a safety decision, where one confident wrong answer can be expensive, set it tight and route refused molecules to an experiment. The threshold is a policy decision informed by the model, not a property of the model.
