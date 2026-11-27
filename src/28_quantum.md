---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Quantum Chemistry with Python

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 12, 13, 16, 17 and an introductory quantum chemistry course.
```

Quantum mechanics explains why bonds form, why molecules are coloured and why reactions have barriers. This chapter moves from textbook models solved numerically to real *ab initio* calculations with **PySCF** {cite:p}`sun2018`.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
```

## Solving the Schrödinger equation on a grid

Discretising $-\frac{\hbar^2}{2m}\psi'' + V\psi = E\psi$ with finite differences turns it into a matrix eigenvalue problem. In atomic units ($\hbar = m_e = 1$):

```{code-cell} ipython3
def solve_1d(V, x, n_states=6, mass=1.0):
    dx = x[1] - x[0]
    diag = 1 / (mass * dx**2) + V
    off = -0.5 / (mass * dx**2) * np.ones(len(x) - 1)
    E, psi = eigh_tridiagonal(diag, off, select="i", select_range=(0, n_states - 1))
    return E, psi / np.sqrt(dx)

def plot_states(x, V, E, psi, title, scale=0.6, ax=None):
    ax = ax or plt.gca()
    ax.plot(x, V, "k", lw=2)
    for i, (e, p) in enumerate(zip(E, psi.T)):
        c = plt.cm.plasma(i / len(E))
        ax.axhline(e, color=c, lw=0.6, ls=":")
        ax.fill_between(x, e, e + scale * p, color=c, alpha=0.6)
    ax.set(title=title, xlabel="x / bohr", ylabel="E / hartree")

x = np.linspace(-6, 6, 2000)
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
V_box = np.where(np.abs(x) < 3, 0, 50.0)
E, psi = solve_1d(V_box, x); plot_states(x, V_box, E, psi, "particle in a box", ax=axes[0]); axes[0].set_ylim(-0.1, 3)
V_ho = 0.5 * 0.5**2 * x**2
E, psi = solve_1d(V_ho, x); plot_states(x, V_ho, E, psi, "harmonic oscillator", scale=0.5, ax=axes[1]); axes[1].set_ylim(0, 3.5)
De, a, xe = 1.5, 0.7, -1.5
V_morse = De * (1 - np.exp(-a * (x - xe)))**2
E, psi = solve_1d(np.minimum(V_morse, 4), x); plot_states(x, np.minimum(V_morse, 4), E, psi, "Morse oscillator", scale=0.4, ax=axes[2])
axes[2].set_ylim(0, 2)
plt.tight_layout(); plt.show()
print("HO levels in units of ħω:", (solve_1d(V_ho, x)[0][:5] / 0.5).round(4))
```

The harmonic levels are evenly spaced at $(n + \frac{1}{2})\hbar\omega$; the Morse levels crowd together towards dissociation, just as vibrational overtones do in real IR spectra.

## Conjugated dyes: particle in a box meets colour

The π electrons of a cyanine dye behave like particles in a one-dimensional box of length *L*. The HOMO→LUMO transition gives

$$\lambda = \frac{8 m_e c L^2}{h (N + 1)}$$

```{code-cell} ipython3
from scipy import constants as C
for k in range(1, 5):                    # number of CH=CH units
    N = 2 * k + 4                        # π electrons
    L = (2 * k + 4) * 1.39e-10           # box length: bonds × 1.39 Å
    lam = 8 * C.m_e * C.c * L**2 / (C.h * (N + 1))
    print(f"k = {k}: N = {N:2d} π electrons, λ = {lam * 1e9:5.0f} nm")
```

Each added double bond shifts absorption to longer wavelength by about 100 nm, in line with experiment.

## Hydrogen atom orbitals

```{code-cell} ipython3
from scipy.special import sph_harm_y, genlaguerre
from math import factorial

def psi_nlm(n, l, m, r, theta, phi):
    rho = 2 * r / n
    norm = np.sqrt((2 / n) ** 3 * factorial(n - l - 1) / (2 * n * factorial(n + l)))
    R = norm * np.exp(-rho / 2) * rho**l * genlaguerre(n - l - 1, 2 * l + 1)(rho)
    return R * sph_harm_y(l, m, theta, phi)

grid = np.linspace(-20, 20, 300)
X, Z = np.meshgrid(grid, grid)
r = np.hypot(X, Z); theta = np.arccos(np.divide(Z, r, out=np.zeros_like(Z), where=r > 0)); phi = np.where(X >= 0, 0, np.pi)

fig, axes = plt.subplots(2, 4, figsize=(13, 6.5))
for ax, (n, l, m, label) in zip(axes.flat, [(1, 0, 0, "1s"), (2, 0, 0, "2s"), (2, 1, 0, "2p$_z$"), (3, 0, 0, "3s"),
                                           (3, 1, 0, "3p$_z$"), (3, 2, 0, "3d$_{z^2}$"), (4, 2, 0, "4d$_{z^2}$"), (4, 3, 0, "4f$_{z^3}$")]):
    psi = psi_nlm(n, l, m, r, theta, phi).real
    lim = np.abs(psi).max() * 0.6
    ax.imshow(psi, cmap="RdBu_r", vmin=-lim, vmax=lim, extent=[-20, 20, -20, 20])
    ax.set_title(label); ax.axis("off")
fig.suptitle("Hydrogen orbitals in the xz-plane (red/blue = phase)")
plt.tight_layout(); plt.show()
```

## Hückel theory for any conjugated molecule

We build the Hückel matrix directly from an RDKit molecule, then draw the HOMO and LUMO coefficients on the structure.

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG

def huckel(smiles):
    m = Chem.MolFromSmiles(smiles)
    Chem.Kekulize(m)
    conj = [a.GetIdx() for a in m.GetAtoms() if a.GetHybridization() == Chem.HybridizationType.SP2 and a.GetSymbol() == "C"]
    idx = {a: i for i, a in enumerate(conj)}
    H = np.zeros((len(conj), len(conj)))
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in idx and j in idx:
            H[idx[i], idx[j]] = H[idx[j], idx[i]] = -1.0
    E, Cmat = np.linalg.eigh(H)
    return m, conj, E, Cmat

def draw_mo(m, conj, coeffs, legend):
    colours, radii = {}, {}
    for a, c in zip(conj, coeffs):
        colours[a] = (0.9, 0.28, 0.48) if c > 0 else (0.06, 0.62, 0.54)
        radii[a] = 0.15 + 1.1 * abs(c)
    d = rdMolDraw2D.MolDraw2DSVG(300, 220)
    rdMolDraw2D.PrepareAndDrawMolecule(d, m, highlightAtoms=list(conj), highlightAtomColors=colours,
                                       highlightAtomRadii=radii, highlightBonds=[], legend=legend)
    d.FinishDrawing()
    return d.GetDrawingText()

for name, smi in [("hexatriene", "C=CC=CC=C"), ("naphthalene", "c1ccc2ccccc2c1"), ("azulene", "c1ccc2cccc2cc1")]:
    m, conj, E, Cm = huckel(smi)
    homo = len(conj) // 2 - 1
    print(f"{name}: HOMO–LUMO gap = {E[homo + 1] - E[homo]:.3f} |β|")
    display(SVG(draw_mo(m, conj, Cm[:, homo], f"{name} HOMO")))
    display(SVG(draw_mo(m, conj, Cm[:, homo + 1], f"{name} LUMO")))
```

Circle size shows the coefficient magnitude; colour shows its sign.

## Hartree–Fock and DFT with PySCF

Now a real calculation: water with a Gaussian basis set.

```{code-cell} ipython3
from pyscf import gto, scf, dft

mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", basis="6-31g*", verbose=0)
mf = scf.RHF(mol).run()
ks = dft.RKS(mol, xc="b3lyp").run()
print(f"RHF energy   {mf.e_tot:.6f} Eh")
print(f"B3LYP energy {ks.e_tot:.6f} Eh")
print("dipole (D):", np.round(mf.dip_moment(verbose=0), 3))
```

### Orbital energy diagram

```{code-cell} ipython3
labels = mol.ao_labels()
e = mf.mo_energy * 27.2114
nocc = mol.nelectron // 2
fig, ax = plt.subplots(figsize=(4, 5))
for i, ei in enumerate(e[:10]):
    occ = i < nocc
    ax.hlines(ei, 0, 1, color="#e5487a" if occ else "#0f9d8a", lw=3)
    if occ:
        ax.text(0.5, ei + 0.4, "↑↓", ha="center", fontsize=11)
ax.set(ylim=(-40, 15), xticks=[], ylabel="orbital energy / eV", title="H₂O RHF/6-31G* MOs")
ax.text(1.05, e[nocc - 1], "HOMO", va="center"); ax.text(1.05, e[nocc], "LUMO", va="center")
plt.show()
```

### A potential energy curve: H₂

```{code-cell} ipython3
Rs = np.linspace(0.4, 3.0, 25)
E_rhf, E_uhf = [], []
for R_ in Rs:
    m = gto.M(atom=f"H 0 0 0; H 0 0 {R_}", basis="cc-pvdz", verbose=0)
    E_rhf.append(scf.RHF(m).run().e_tot)
    uhf = scf.UHF(m)
    dm = uhf.get_init_guess(); dm[0][0, 0] += 0.3; dm[1][0, 0] -= 0.3      # break spin symmetry
    E_uhf.append(uhf.run(dm).e_tot)
plt.plot(Rs, E_rhf, "o-", color="#6c4ad6", label="RHF")
plt.plot(Rs, E_uhf, "s-", color="#f2a93b", label="UHF")
plt.axhline(-1.0, ls="--", color="k", label="2 H atoms (exact)")
plt.xlabel("R / Å"); plt.ylabel("E / hartree"); plt.legend(); plt.title("Dissociating H₂: why RHF fails")
plt.show()
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Restricted Hartree–Fock forces both electrons into the same spatial orbital, so H₂ dissociates into a mixture of H⁺ + H⁻ and H + H. This **static correlation** problem is why bond breaking needs multireference or carefully broken-symmetry methods.
```

## Exercises

**★ 1.** Change the box width in the particle-in-a-box solver and verify that E ∝ 1/L².

**★★ 2.** Use the Hückel function to compare the HOMO–LUMO gaps of benzene, naphthalene, anthracene and tetracene. Relate the trend to their colours.

**★★★ 3.** Optimise the geometry of water with PySCF (`pyscf.geomopt`) at HF/6-31G* and B3LYP/6-31G*, and compare the bond angles with experiment (104.5°).
