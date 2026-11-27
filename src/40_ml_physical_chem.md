---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Physical and Quantum Chemistry: ML Potentials

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 28, 29 and Part V.
```

Quantum chemistry is accurate but slow; force fields are fast but approximate. **Machine-learned potentials** aim for both: train on quantum-mechanical energies and forces, then run at near-force-field speed. The ideas go back to neural-network potentials {cite:p}`behler2007`, Coulomb-matrix learning of atomisation energies {cite:p}`rupp2012`, Δ-learning {cite:p}`ramakrishnan2015` and architectures such as ANI {cite:p}`smith2017` and SchNet {cite:p}`schutt2018`, trained on datasets like QM9 {cite:p}`ramakrishnan2014`. A **potential energy surface** (PES) gives the energy of a group of atoms as a function of their positions; its minima are stable structures, its saddle points are transition states, and its slopes are the forces that drive molecular dynamics (Chapter 29). A quantum calculation (Chapter 28) gives one point on the PES at a time, often taking minutes to hours, while a simulation needs millions of points. A machine-learned potential is a regression model (Chapter 31) whose input is a description of the atomic positions and whose output is the energy, trained on a few hundred or thousand quantum calculations and then evaluated in microseconds. This chapter learns a one-dimensional PES, computes a small dataset of molecular energies with PySCF, compares two ways of describing molecules, tries Δ-learning, and runs dynamics on a fitted potential.

The cells in this chapter run real quantum-chemistry calculations, so they take a few minutes.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf import gto, scf, dft
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.metrics import mean_absolute_error
rng = np.random.default_rng(0)
HARTREE_KCAL = 627.5095
```

## Learning a potential energy surface

Start with the simplest possible case: the bond-stretch curve of HF, computed with density functional theory, learned from a handful of points. The model is **kernel ridge regression** (KRR), the workhorse of early ML potentials. A **kernel** measures how similar two inputs are; the RBF kernel, $\exp(-\gamma |x - x'|^2)$, is 1 for identical inputs and falls towards 0 as they move apart, with $\gamma$ setting how quickly (Chapter 35 used the same idea in Gaussian processes). KRR predicts a new point as a weighted sum of kernel similarities to the training points, with the weights fitted so the training data are reproduced; `alpha` adds a little regularisation. With a tiny regularisation, it passes almost exactly through the training data and interpolates smoothly between them:

```{code-cell} ipython3
toy = KernelRidge(alpha=1e-8, kernel="rbf", gamma=1.0).fit([[0.0], [1.0], [2.0]], [0.0, 1.0, 4.0])
print(toy.predict([[0.0], [1.5], [2.0]]).round(2))
```

Below, 30 B3LYP/6-31G* energies along the bond are computed, and only 7 of them are used to train the model.

```{code-cell} ipython3
def hf_energy(r, method="b3lyp"):
    m = gto.M(atom=f"H 0 0 0; F 0 0 {r}", basis="6-31g*", verbose=0)
    return (dft.RKS(m, xc="b3lyp") if method == "b3lyp" else scf.RHF(m)).run().e_tot

R = np.linspace(0.7, 2.4, 30)
E = np.array([hf_energy(r) for r in R])
E_rel = (E - E.min()) * HARTREE_KCAL

train_idx = np.array([0, 4, 7, 10, 14, 20, 29])
krr = KernelRidge(alpha=1e-8, kernel="rbf", gamma=1.5).fit(R[train_idx, None], E_rel[train_idx])
fine = np.linspace(0.7, 2.4, 300)
plt.plot(fine, krr.predict(fine[:, None]), color="#6c4ad6", lw=2, label="kernel ridge model")
plt.plot(R, E_rel, "k:", lw=1, label="DFT reference")
plt.plot(R[train_idx], E_rel[train_idx], "o", color="#e5487a", ms=9, label="training points")
plt.xlabel("r(H–F) / Å"); plt.ylabel("E / kcal mol$^{-1}$"); plt.legend()
plt.title(f"7 points reproduce the curve to {mean_absolute_error(E_rel, krr.predict(R[:, None])):.2f} kcal/mol")
plt.show()
```

## A molecular dataset: energies of small molecules

Now the real task: predict energies of **different molecules** from their structures. We compute a small QM9-style dataset ourselves, so the book needs no external download. Each molecule's 3D geometry comes from RDKit (Chapter 23), and its energy from a Hartree–Fock calculation with PySCF (Chapter 28). Total energies are dominated by the core electrons of each atom and are not chemically interesting, so the target is the **atomisation energy**: the molecule's energy minus the energies of its separated atoms, a negative number whose size measures the total bonding. For H₂ it is twice the H-atom energy subtracted from the molecular energy:

```{code-cell} ipython3
e_h2 = scf.RHF(gto.M(atom="H 0 0 0; H 0 0 0.74", basis="6-31g", verbose=0)).run().e_tot
e_h = scf.UHF(gto.M(atom="H 0 0 0", basis="6-31g", spin=1, verbose=0)).run().e_tot
print(f"{(e_h2 - 2 * e_h) * HARTREE_KCAL:.1f} kcal/mol")
```

Hartree–Fock underestimates bond energies because it neglects electron correlation; the experimental value for H₂ is about −109 kcal/mol. The atom energies are stored in a dictionary so each element is computed only once, a simple form of caching.

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import AllChem

SMILES = ["C", "CC", "CCC", "CCCC", "CC(C)C", "CCCCC", "CC(C)CC", "CC(C)(C)C", "C=C", "C=CC", "C=CC=C", "C#C",
          "C#CC", "CO", "CCO", "CCCO", "CC(C)O", "COC", "CCOC", "C=O", "CC=O", "CCC=O", "CC(C)=O", "C(=O)O",
          "CC(=O)O", "CCC(=O)O", "N", "CN", "CCN", "CCCN", "CNC", "CC#N", "CCC#N", "C1CC1", "C1CCC1", "C1CCCC1",
          "C1CO1", "C1CCO1", "OCC O".replace(" ", ""), "OCCO", "NCC O".replace(" ", ""), "NCCO", "CSC", "CS",
          "CCS", "ClC", "ClCC", "FC", "FCC", "C1CCNC1", "C1COC1", "CC(N)=O", "CNC=O", "OC=O", "CC(C)N"]

def geometry(smi, seed=1):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    p = AllChem.ETKDGv3(); p.randomSeed = seed
    AllChem.EmbedMolecule(m, p); AllChem.MMFFOptimizeMolecule(m)
    conf = m.GetConformer()
    atoms = [(a.GetSymbol(), *conf.GetAtomPosition(a.GetIdx())) for a in m.GetAtoms()]
    return m, atoms

ATOM_E = {}          # single-atom reference energies, to get atomisation energies
def atom_energy(sym):
    if sym not in ATOM_E:
        spin = {"H": 1, "C": 2, "N": 3, "O": 2, "F": 1, "S": 2, "Cl": 1}[sym]
        m = gto.M(atom=f"{sym} 0 0 0", basis="6-31g", spin=spin, verbose=0)
        ATOM_E[sym] = scf.UHF(m).run().e_tot
    return ATOM_E[sym]

rows = []
for smi in SMILES:
    try:
        mol, atoms = geometry(smi)
        spec = "; ".join(f"{s} {x:.4f} {y:.4f} {z:.4f}" for s, x, y, z in atoms)
        e = scf.RHF(gto.M(atom=spec, basis="6-31g", verbose=0)).run().e_tot
        atomisation = (e - sum(atom_energy(s) for s, *_ in atoms)) * HARTREE_KCAL
        rows.append({"smiles": smi, "atoms": atoms, "E_total": e, "E_atomisation": atomisation, "n_atoms": len(atoms)})
    except Exception as err:
        print("skipped", smi, err)
qm = pd.DataFrame(rows)
print(len(qm), "molecules;  atomisation energies from", f"{qm.E_atomisation.min():.0f}", "to",
      f"{qm.E_atomisation.max():.0f}", "kcal/mol")
```

## Representation 1: the Coulomb matrix

$$M_{ij} = \begin{cases} 0.5\,Z_i^{2.4} & i = j\\ Z_iZ_j / |\mathbf R_i - \mathbf R_j| & i \ne j\end{cases}$$

Sorting the eigenvalues makes it invariant to atom ordering, rotation and translation. A representation for an ML potential must not change when the molecule is moved, rotated or its atoms are renumbered, because the energy does not change. The **Coulomb matrix** {cite:p}`rupp2012` has one row and column per atom: each off-diagonal element is the Coulomb repulsion between two nuclear charges, and each diagonal element is a fitted function of the atom's own charge. It depends only on distances, so it ignores translation and rotation, and its **eigenvalues** do not depend on the order of the atoms. Molecules have different numbers of atoms, so the sorted eigenvalue lists are padded with zeros to the same length. For H₂ the matrix is 2 × 2:

```{code-cell} ipython3
r_bohr = 0.74 / 0.529                      # bond length in bohr
M_h2 = np.array([[0.5, 1 / r_bohr], [1 / r_bohr, 0.5]])
print(M_h2.round(3), np.linalg.eigvalsh(M_h2).round(3))
```

```{code-cell} ipython3
Z = {"H": 1, "C": 6, "N": 7, "O": 8, "F": 9, "S": 16, "Cl": 17}
max_atoms = qm.n_atoms.max()

def coulomb_eigs(atoms):
    n = len(atoms)
    pos = np.array([[x, y, z] for _, x, y, z in atoms])
    zs = np.array([Z[s] for s, *_ in atoms], dtype=float)
    d = np.linalg.norm(pos[:, None] - pos[None, :], axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        M = np.where(np.eye(n, dtype=bool), 0.5 * zs ** 2.4, np.outer(zs, zs) / d)
    M = np.nan_to_num(M)
    ev = np.sort(np.linalg.eigvalsh(M))[::-1]
    return np.pad(ev, (0, max_atoms - n))

X_cm = np.array([coulomb_eigs(a) for a in qm.atoms])
y = qm.E_atomisation.values
print(X_cm.shape)
```

## Representation 2: a simple bond-counting (atom-in-molecule) model

A purely additive model over atom and bond types is the physicist's baseline. Chemists have long known that bond energies are roughly transferable: a C–H bond contributes about the same energy in methane as in hexane. A model that counts each type of atom and bond and fits one energy per type with linear regression captures this idea directly, and it is the baseline any more sophisticated representation must beat. Each bond is labelled by its two elements in alphabetical order and its bond order, so ethanol gets counts such as `C-H_1` and `C-O_1`.

```{code-cell} ipython3
def bond_counts(smi):
    m = Chem.AddHs(Chem.MolFromSmiles(smi))
    feats = {f"n_{s}": 0 for s in Z}
    for a in m.GetAtoms():
        feats[f"n_{a.GetSymbol()}"] = feats.get(f"n_{a.GetSymbol()}", 0) + 1
    for b in m.GetBonds():
        key = "-".join(sorted([b.GetBeginAtom().GetSymbol(), b.GetEndAtom().GetSymbol()])) + f"_{int(b.GetBondTypeAsDouble())}"
        feats[key] = feats.get(key, 0) + 1
    feats["n_rings"] = Chem.rdMolDescriptors.CalcNumRings(m)
    return feats

bc = pd.DataFrame([bond_counts(s) for s in qm.smiles]).fillna(0)
X_bc = bc.values.astype(float)
print(X_bc.shape, "bond-count features")
```

## Comparing representations

Both representations are tested with the same 5-fold cross-validation. Kernel ridge has two settings, the regularisation `alpha` and the kernel width `gamma`, and `GridSearchCV` chooses them by an inner cross-validation on each training fold, so the outer test fold is never used for tuning. The Laplacian kernel, $\exp(-\gamma |x - x'|_1)$, is a common choice for Coulomb-matrix features.

```{code-cell} ipython3
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

cv = KFold(5, shuffle=True, random_state=0)

def krr(gammas=(1e-4, 1e-3, 1e-2, 1e-1, 1.0)):
    """Kernel ridge with the kernel width chosen by inner cross-validation."""
    from sklearn.model_selection import GridSearchCV
    return make_pipeline(StandardScaler(),
                         GridSearchCV(KernelRidge(kernel="laplacian"),
                                      {"alpha": [1e-6, 1e-4, 1e-2], "gamma": list(gammas)}, cv=3))

setups = {"Coulomb eigenvalues + KRR": (X_cm, krr()),
          "bond counts + ridge": (X_bc, make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-3, 3, 20)))),
          "bond counts + KRR": (X_bc, krr())}

preds = {}
for name, (Xf, model) in setups.items():
    p = cross_val_predict(model, Xf, y, cv=cv)
    preds[name] = p
    print(f"{name:30s} MAE = {mean_absolute_error(y, p):7.2f} kcal/mol")
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
for ax, (name, p), c in zip(axes, preds.items(), ["#6c4ad6", "#f2a93b", "#0f9d8a"]):
    ax.scatter(y, p, s=40, color=c, edgecolor="k")
    lims = [y.min() - 50, y.max() + 50]
    ax.plot(lims, lims, "k--"); ax.set(xlim=lims, ylim=lims, aspect="equal", xlabel="HF atomisation energy / kcal/mol",
                                       ylabel="predicted", title=f"{name}\nMAE = {mean_absolute_error(y, p):.1f}")
plt.tight_layout(); plt.show()
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Atomisation energies span thousands of kcal/mol, so an R² near 1 is easy and meaningless. Judge a potential by its **MAE against chemical accuracy** (1 kcal/mol), and by whether it predicts *differences* between similar molecules.
```

## Δ-learning: predicting the correction, not the energy

Instead of learning the expensive energy directly, learn the **difference** between a cheap and an expensive method {cite:p}`ramakrishnan2015`. Here the cheap method is the Hartree–Fock calculation we already did and the expensive one is B3LYP/6-31G*; the correction is mostly the (smooth, local) correlation energy, so it is an easier function to learn than the energy itself. The prediction is then the cheap result plus the learned correction. With made-up numbers:

```{code-cell} ipython3
cheap_demo, true_demo = np.array([-100.0, -200.0, -300.0]), np.array([-120.0, -235.0, -352.0])
print("corrections to learn:", true_demo - cheap_demo)   # much smaller range than the energies themselves
```

```{code-cell} ipython3
expensive = []
for atoms in qm.atoms:
    spec = "; ".join(f"{s} {x:.4f} {y:.4f} {z:.4f}" for s, x, y, z in atoms)
    expensive.append(dft.RKS(gto.M(atom=spec, basis="6-31g*", verbose=0), xc="b3lyp").run().e_tot)
expensive = np.array(expensive)
cheap = qm.E_total.values                     # the HF/6-31G energies computed above

# Work with atomisation energies so that both targets are on the same, chemically meaningful scale
ref_cheap = np.array([sum(atom_energy(a[0]) for a in atoms) for atoms in qm.atoms])
y_expensive = (expensive - ref_cheap) * HARTREE_KCAL          # the expensive target
y_cheap = (cheap - ref_cheap) * HARTREE_KCAL                  # a cheap, biased estimate of it
delta = y_expensive - y_cheap                                 # the correction to learn

direct = cross_val_predict(krr(), X_bc, y_expensive, cv=cv)
d_pred = cross_val_predict(krr(), X_bc, delta, cv=cv)
print(f"cheap method alone (HF/6-31G)     MAE = {mean_absolute_error(y_expensive, y_cheap):8.2f} kcal/mol")
print(f"direct learning of the target     MAE = {mean_absolute_error(y_expensive, direct):8.2f} kcal/mol")
print(f"Δ-learning (cheap + correction)   MAE = {mean_absolute_error(y_expensive, y_cheap + d_pred):8.2f} kcal/mol")
plt.scatter(delta, d_pred, s=40, color="#e5487a", edgecolor="k")
plt.plot([delta.min(), delta.max()], [delta.min(), delta.max()], "k--")
plt.xlabel("true correction Δ / kcal/mol"); plt.ylabel("predicted"); plt.title("Δ-learning is an easier target")
plt.show()
```

```{admonition} Read your own results honestly
:class: warning
In this small example Δ-learning does **not** beat direct learning. The reason is instructive: our bond-count features are almost a perfect additive basis for an additive quantity, so a linear model already captures the energy, and the correction inherits the same size dependence. Δ-learning pays off when the cheap method removes most of the variation that the features cannot express, for example when the geometries differ but the composition does not, or when the target is a reaction energy rather than an atomisation energy. Try it with the Coulomb-matrix representation, or with conformers of one molecule, and the ordering reverses.
```

## A machine-learned force field for MD

For dynamics we need **forces**, which are derivatives of the energy. Here we fit a potential to a diatomic and run dynamics with it, comparing against the quantum reference. The force on a bond is minus the slope of the energy, $F = -dE/dr$: it pulls the atoms together when the bond is stretched and pushes them apart when it is compressed. A **Morse potential**, $D_e(1 - e^{-a(r - r_e)})^2$, is a physically motivated function with three meaningful parameters: the well depth $D_e$, the equilibrium bond length $r_e$ and the stiffness $a$. Fitting it with `curve_fit` (Chapter 16) gives an analytic energy whose derivative is the force. A finite-difference check confirms that the analytic and numerical slopes agree:

```{code-cell} ipython3
morse_demo = lambda r: 100 * (1 - np.exp(-2 * (r - 1.0))) ** 2
h = 1e-5
print((morse_demo(1.2 + h) - morse_demo(1.2 - h)) / (2 * h),
      2 * 100 * 2 * np.exp(-2 * 0.2) * (1 - np.exp(-2 * 0.2)))   # numerical vs analytic dE/dr
```

Newton's equations of motion for the bond length are then integrated with `solve_ivp` (Chapter 16), starting from a stretched bond at rest.

```{code-cell} ipython3
from scipy.optimize import curve_fit
from scipy.integrate import solve_ivp

def morse(r, De, a, re, E0):
    return De * (1 - np.exp(-a * (r - re))) ** 2 + E0

popt, _ = curve_fit(morse, R, E_rel, p0=[140, 2.0, 0.93, 0])
De, a, re, E0 = popt
force = lambda r: -2 * De * a * np.exp(-a * (r - re)) * (1 - np.exp(-a * (r - re)))
mu = 1.008 * 18.998 / (1.008 + 18.998) * 1822.9          # reduced mass in electron masses
conv = 1 / (HARTREE_KCAL)                                 # kcal/mol -> hartree

def eom(t, s):
    r, v = s
    return [v, force(r) * conv / mu * 0.529]              # crude units: Å, atomic time

sol = solve_ivp(eom, (0, 4000), [1.15, 0.0], dense_output=True, rtol=1e-8)
tt = np.linspace(0, 4000, 2000)
r_t = sol.sol(tt)[0]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(R, E_rel, "ko", ms=4, label="DFT"); axes[0].plot(fine, morse(fine, *popt), color="#6c4ad6", label="fitted Morse")
axes[0].legend(); axes[0].set(xlabel="r / Å", ylabel="E / kcal/mol", title=f"De = {De:.0f} kcal/mol, re = {re:.3f} Å")
axes[1].plot(tt, r_t, color="#0f9d8a"); axes[1].set(xlabel="time (arb. units)", ylabel="r(H–F) / Å", title="Vibration on the learned surface")
plt.tight_layout(); plt.show()
from scipy.signal import find_peaks
peaks, _ = find_peaks(r_t)
period_au = np.mean(np.diff(tt[peaks]))
print(f"Vibrational period on the learned surface: {period_au:.0f} (arbitrary time units)")
print(f"Harmonic ω from the Morse fit: a*sqrt(2*De/mu) = {a * np.sqrt(2 * De * conv / mu):.5f} a.u.")
```

We have kept the units deliberately simple, so these numbers are illustrative rather than spectroscopic. The workflow, fit a potential to quantum data and then integrate the equations of motion, is exactly how machine-learned interatomic potentials drive MD at DFT accuracy and force-field cost.

```{admonition} Summary
:class: tip
- ML potentials learn the potential energy surface from quantum data and evaluate it at force-field speed.
- Kernel ridge regression interpolates energies from a few training points.
- Learn atomisation energies, not total energies, and judge models against 1 kcal/mol.
- Representations must be invariant to translation, rotation and atom order: Coulomb-matrix eigenvalues, bond counts.
- Δ-learning predicts the correction from a cheap to an expensive method; check whether it actually helps.
- Forces are minus the energy gradient; a fitted potential drives molecular dynamics.
```

## Exercises

**★ 1.** Fit the HF curve with 3, 5 and 10 training points. How many are needed for 1 kcal/mol accuracy?

**★★ 2.** Add second-nearest-neighbour counts to the bond-count features and see whether the MAE improves.

**★★★ 3.** Compute energies for three conformers of butane and check whether either representation can rank them. Why does the Coulomb matrix succeed where bond counts fail?
