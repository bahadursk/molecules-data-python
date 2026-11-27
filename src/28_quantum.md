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

Quantum mechanics explains why bonds form, why molecules are coloured and why reactions have barriers. Its central equation, the time-independent **Schrödinger equation** $\hat{H}\psi = E\psi$, says that the allowed energies $E$ of a system and its **wavefunctions** $\psi$ are the solutions of an eigenvalue problem: the Hamiltonian operator $\hat{H}$ (kinetic plus potential energy) acting on $\psi$ gives back $\psi$ multiplied by a number. The square of the wavefunction, $|\psi|^2$, is the probability density of finding the particle. Only a few textbook systems can be solved with pen and paper; everything else is solved numerically, and the trick is always to turn the equation into a matrix whose eigenvalues are the energies. **Quantum chemistry** applies this to molecules to predict structures, energies, spectra and reactivity from first principles. This chapter moves from textbook models solved numerically to real *ab initio* calculations with **PySCF** {cite:p}`sun2018`.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh_tridiagonal
```

## Solving the Schrödinger equation on a grid

The key numerical idea is to represent $\psi$ by its values on a fine grid of points. The second derivative at each point can then be approximated from its neighbours, $\psi''(x_i) \approx (\psi_{i+1} - 2\psi_i + \psi_{i-1})/\Delta x^2$, a *finite difference*. With this replacement the Schrödinger equation becomes a set of linear equations linking each point to its neighbours, which is a matrix equation $\mathbf{H}\boldsymbol{\psi} = E\boldsymbol{\psi}$. Each point couples only to its two neighbours, so the matrix is *tridiagonal* (non-zero only on the diagonal and next to it), and `scipy.linalg.eigh_tridiagonal` finds its lowest eigenvalues quickly. **Atomic units** set $\hbar$, the electron mass and the elementary charge to 1, so energies come out in hartrees and lengths in bohr. Discretising $-\frac{\hbar^2}{2m}\psi'' + V\psi = E\psi$ with finite differences turns it into a matrix eigenvalue problem. In atomic units ($\hbar = m_e = 1$):

The idea fits in a few lines for the simplest case, a particle in a box of length 1 with infinitely high walls, whose exact energies are $E_n = n^2\pi^2/2$ in atomic units. Building the matrix as a dense array makes its structure visible:

```{code-cell} ipython3
n_pts, L_box = 200, 1.0
dx_box = L_box / (n_pts + 1)
H_box = (np.diag(np.full(n_pts, 1.0)) - 0.5 * np.diag(np.ones(n_pts - 1), 1)
         - 0.5 * np.diag(np.ones(n_pts - 1), -1)) / dx_box**2
print("numerical:", np.linalg.eigvalsh(H_box)[:3].round(3))
print("exact:    ", (np.arange(1, 4) ** 2 * np.pi**2 / 2).round(3))
```

The agreement is excellent. A general solver accepts any potential $V(x)$ on the grid, which lets us compare a box with soft walls, a harmonic oscillator and a Morse oscillator, the more realistic model of a chemical bond that dissociates at large separation:

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

The particle in a box is not only a teaching model. In a conjugated molecule the π electrons are delocalised along the chain and move almost freely within it, so the chain acts as a one-dimensional box. The box levels are $E_n = n^2h^2/(8m_eL^2)$, and with $N$ π electrons filling the levels two by two, the lowest-energy absorption promotes an electron from the highest occupied level $n = N/2$ to the next. The longer the box, the closer the levels and the longer the wavelength absorbed, which is why extending conjugation shifts a dye's colour towards red. The π electrons of a cyanine dye behave like particles in a one-dimensional box of length *L*. The HOMO→LUMO transition gives

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

The hydrogen atom is the one real atom with an exact solution. Each **orbital** is labelled by three quantum numbers, $n$, $l$ and $m$, and factorises into a radial part $R_{nl}(r)$, which depends on the distance from the nucleus and is built from Laguerre polynomials, and an angular part, the spherical harmonic $Y_l^m(\theta, \phi)$, which gives the familiar s, p, d and f shapes. Both special functions are in `scipy.special`. Plotting a slice through the xz-plane with a diverging colormap shows the **nodes**, surfaces where $\psi = 0$ and the phase changes sign: an orbital has $n - 1$ nodes in total, $l$ of them angular.

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

**Hückel theory** is the simplest molecular-orbital model of π systems. Each sp² carbon contributes one p orbital, and the Hamiltonian matrix is built from just two numbers: $\alpha$ on the diagonal (the energy of an isolated p orbital, set to 0) and $\beta$ between bonded atoms (a negative interaction energy, set to −1). Non-bonded pairs get 0, so the Hückel matrix is simply the negative of the adjacency matrix of the π framework (Chapter 19). Its eigenvalues are the orbital energies in units of $|\beta|$, and its eigenvectors give each atom's coefficient in each molecular orbital. For ethene there are two orbitals, bonding and antibonding:

```{code-cell} ipython3
H_ethene = np.array([[0.0, -1.0], [-1.0, 0.0]])
E_eth, C_eth = np.linalg.eigh(H_ethene)
print("energies (units of |β|):", E_eth)
print("bonding orbital coefficients:", C_eth[:, 0].round(3))
```

The bonding orbital lies at $\alpha + \beta$ with coefficients of equal size, $1/\sqrt{2}$, on both carbons (the overall sign of an eigenvector is arbitrary, so both may come out negative). We build the Hückel matrix directly from an RDKit molecule, then draw the HOMO and LUMO coefficients on the structure.

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

Circle size shows the coefficient magnitude; colour shows its sign. Naphthalene, with ten π electrons in an aromatic framework, has a larger gap than the open-chain hexatriene, a sign of aromatic stabilisation. Azulene, an isomer of naphthalene built from a five- and a seven-membered ring, has a much smaller gap, which is why it is blue while naphthalene is colourless.

## Hartree–Fock and DFT with PySCF

Real molecules have many interacting electrons, and the Schrödinger equation must be approximated. Two families of methods dominate. **Hartree–Fock** (HF) treats each electron as moving in the average field of all the others, so the many-electron problem becomes a set of one-electron orbital equations solved *self-consistently*: guess the orbitals, compute the average field, solve for new orbitals, and repeat until nothing changes (the **self-consistent field**, SCF). HF neglects the way electrons dodge each other instantaneously (**electron correlation**). **Density functional theory** (DFT) instead works with the electron density and includes correlation approximately through an exchange–correlation *functional*, such as the widely used B3LYP; it usually gives better energies at similar cost.

Both methods expand the orbitals in a **basis set**, a fixed collection of Gaussian-shaped functions centred on each atom. Larger basis sets are more accurate and more expensive; `6-31g*` is a common small basis for organic molecules. PySCF builds a molecule with `gto.M` from coordinates (in ångströms) and a basis, and `scf.RHF(mol).run()` performs a restricted Hartree–Fock calculation, in which each spatial orbital holds two electrons of opposite spin. The smallest possible example is the hydrogen molecule:

```{code-cell} ipython3
from pyscf import gto, scf, dft

h2 = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
print(f"H2 RHF/STO-3G energy: {scf.RHF(h2).run().e_tot:.4f} hartree")
```

Energies are reported in hartrees, and one hartree is 2625.5 kJ mol⁻¹, so differences of a few millihartree are chemically significant. Now a real calculation: water with a Gaussian basis set.

```{code-cell} ipython3
mol = gto.M(atom="O 0 0 0.1173; H 0 0.7572 -0.4692; H 0 -0.7572 -0.4692", basis="6-31g*", verbose=0)
mf = scf.RHF(mol).run()
ks = dft.RKS(mol, xc="b3lyp").run()
print(f"RHF energy   {mf.e_tot:.6f} Eh")
print(f"B3LYP energy {ks.e_tot:.6f} Eh")
print("dipole (D):", np.round(mf.dip_moment(verbose=0), 3))
```

The B3LYP energy is lower than the HF energy because DFT includes electron correlation. The dipole moment, computed from the electron density, comes out somewhat larger than the experimental 1.85 D, a known tendency of HF with small basis sets.

### Orbital energy diagram

The SCF calculation also gives the **molecular orbital energies**, stored in `mf.mo_energy` in hartrees (multiplied by 27.2114 to convert to eV). With 10 electrons, water fills the lowest five orbitals; the highest occupied (HOMO) and lowest unoccupied (LUMO) orbitals control its chemistry and its lowest electronic excitation.

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

A **potential energy curve** gives the energy of a molecule as a function of a bond length, and its minimum is the equilibrium bond length. Computing one is a loop: build the molecule at each separation, run the calculation and store the energy. For H₂ the curve also exposes a famous limitation. **Unrestricted** HF (UHF) allows the two electrons to occupy different spatial orbitals; to let it find that solution, the initial guess density is nudged so that the spin symmetry is broken.

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

```{admonition} Summary
:class: tip
- Finite differences turn the Schrödinger equation into a matrix eigenvalue problem; eigenvalues are energies and eigenvectors wavefunctions.
- Harmonic levels are evenly spaced; Morse levels converge towards dissociation.
- Conjugated chains behave like particles in a box: longer chains absorb at longer wavelengths.
- Hydrogen orbitals are radial functions times spherical harmonics; nodes are where the phase changes.
- Hückel theory: the π Hamiltonian is minus the adjacency matrix; diagonalise for orbital energies and coefficients.
- PySCF runs HF and DFT with Gaussian basis sets; RHF fails at bond breaking, which UHF partly repairs.
```

## Exercises

**★ 1.** Change the box width in the particle-in-a-box solver and verify that E ∝ 1/L².

**★★ 2.** Use the Hückel function to compare the HOMO–LUMO gaps of benzene, naphthalene, anthracene and tetracene. Relate the trend to their colours.

**★★★ 3.** Optimise the geometry of water with PySCF (`pyscf.geomopt`) at HF/6-31G* and B3LYP/6-31G*, and compare the bond angles with experiment (104.5°).
