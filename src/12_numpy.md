---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# NumPy: Arrays for Chemical Data

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Part I.
```

A spectrum is 4 000 absorbances, a trajectory is a million coordinates, an image is a grid of intensities. **NumPy** stores such data in fast, compact **arrays** and lets you compute on whole arrays at once.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)
```

## Creating arrays

```{code-cell} ipython3
masses = np.array([1.008, 12.011, 14.007, 15.999])
print(masses, masses.dtype, masses.shape)

print(np.zeros(3), np.ones((2, 3)), sep="\n")
wavelengths = np.arange(400, 701, 50)          # start, stop, step
T = np.linspace(273.15, 373.15, 5)              # start, stop, number of points
print(wavelengths, T, sep="\n")
```

## Vectorised arithmetic

Operations apply to **every element** without a loop, and run tens to hundreds of times faster.

```{code-cell} ipython3
T_C = np.array([-78.5, 0.0, 25.0, 37.0, 100.0])
T_K = T_C + 273.15
kT = 1.380649e-23 * T_K
print(T_K)
print(np.exp(-1e-20 / kT))       # Boltzmann factors
```

```{code-cell} ipython3
import time
x = np.random.default_rng(0).random(1_000_000)
t0 = time.perf_counter(); s1 = sum(v * v for v in x); t1 = time.perf_counter()
s2 = np.sum(x * x); t2 = time.perf_counter()
print(f"Python loop {1000*(t1-t0):.1f} ms, NumPy {1000*(t2-t1):.2f} ms")
```

## Indexing, slicing and Boolean masks

```{code-cell} ipython3
rng = np.random.default_rng(42)
wn = np.linspace(4000, 400, 3601)                  # IR wavenumbers / cm^-1
peaks = [(3350, 120, 0.6), (2950, 40, 0.5), (1715, 25, 0.9), (1250, 40, 0.4)]
absorb = sum(h * np.exp(-((wn - c) / w) ** 2) for c, w, h in peaks)
absorb += rng.normal(0, 0.01, wn.size)
transmittance = 10 ** -absorb

print(wn[:3], wn[-3:])
carbonyl = (wn > 1650) & (wn < 1800)             # Boolean mask
print("carbonyl region max A:", absorb[carbonyl].max().round(3),
      "at", wn[carbonyl][absorb[carbonyl].argmax()], "cm^-1")
```

```{code-cell} ipython3
fig, ax = plt.subplots(figsize=(8, 3.5))
ax.plot(wn, 100 * transmittance, color="#6c4ad6")
ax.fill_between(wn, 100 * transmittance, 100, where=carbonyl, color="#e5487a", alpha=0.4, label="C=O stretch")
ax.fill_between(wn, 100 * transmittance, 100, where=(wn > 3100) & (wn < 3600), color="#0f9d8a", alpha=0.3, label="O–H stretch")
ax.invert_xaxis(); ax.set_xlabel("Wavenumber / cm$^{-1}$"); ax.set_ylabel("% T")
ax.legend(); ax.set_title("Simulated IR spectrum of a hydroxy ketone")
plt.show()
```

## Two-dimensional arrays and broadcasting

Arrays can have any number of dimensions. Rows and columns are indexed `[row, column]`.

```{code-cell} ipython3
# Coordinates of methane (Å)
xyz = np.array([[ 0.000,  0.000,  0.000],
                [ 0.629,  0.629,  0.629],
                [-0.629, -0.629,  0.629],
                [-0.629,  0.629, -0.629],
                [ 0.629, -0.629, -0.629]])
print(xyz.shape, xyz[0], xyz[:, 2])
```

**Broadcasting** stretches arrays of compatible shapes. Subtracting a `(5, 1, 3)` array from a `(1, 5, 3)` array gives all pairwise differences: a full distance matrix in one line.

```{code-cell} ipython3
D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
print(D.round(3))

v1, v2 = xyz[1] - xyz[0], xyz[2] - xyz[0]
angle = np.degrees(np.arccos(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))))
print(f"H–C–H angle = {angle:.2f}° (tetrahedral 109.47°)")
```

```{code-cell} ipython3
plt.imshow(D, cmap="viridis")
plt.colorbar(label="distance / Å")
plt.xticks(range(5), ["C", "H", "H", "H", "H"]); plt.yticks(range(5), ["C", "H", "H", "H", "H"])
plt.title("Methane distance matrix")
plt.show()
```

## Statistics along axes

```{code-cell} ipython3
# Triplicate absorbance readings (rows) at five concentrations (columns)
A = np.array([[0.101, 0.198, 0.305, 0.399, 0.502],
              [0.098, 0.203, 0.298, 0.405, 0.497],
              [0.103, 0.201, 0.301, 0.402, 0.505]])
print("mean:", A.mean(axis=0))
print("std: ", A.std(axis=0, ddof=1))
```

## Linear algebra

### Least-squares calibration line

```{code-cell} ipython3
c = np.array([1, 2, 3, 4, 5]) * 1e-5           # mol/L
y = A.mean(axis=0)
X = np.column_stack([c, np.ones_like(c)])
(slope, intercept), *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"ε·l = {slope:.0f} L/mol, intercept = {intercept:.4f}")
```

### Balancing a chemical equation

Balancing $a\,\mathrm{C_3H_8} + b\,\mathrm{O_2} \rightarrow c\,\mathrm{CO_2} + d\,\mathrm{H_2O}$ is solving a homogeneous linear system: each element gives one row. The null space of the matrix contains the coefficients.

```{code-cell} ipython3
#              C3H8  O2  CO2  H2O
M = np.array([[  3,   0,  -1,   0],   # C
              [  8,   0,   0,  -2],   # H
              [  0,   2,  -2,  -1]])  # O
_, _, Vt = np.linalg.svd(M)
coeffs = Vt[-1] / Vt[-1][np.abs(Vt[-1]) > 1e-9].min()
print(np.round(coeffs, 6))            # 1 C3H8 + 5 O2 -> 3 CO2 + 4 H2O
```

### Hückel molecular orbitals of benzene

In Hückel theory the energies of the π orbitals are the eigenvalues of a matrix with α on the diagonal and β between bonded carbons. Using units where α = 0 and β = −1:

```{code-cell} ipython3
n = 6
H = np.zeros((n, n))
for i in range(n):
    H[i, (i + 1) % n] = H[(i + 1) % n, i] = -1.0
E, C = np.linalg.eigh(H)
print("energies (in |β| units):", E.round(3))

fig, ax = plt.subplots(figsize=(4, 4))
for e in np.unique(E.round(6)):
    k = np.sum(np.isclose(E, e))
    xs = np.linspace(-0.3 * (k - 1), 0.3 * (k - 1), k)
    for x0 in xs:
        ax.hlines(e, x0 - 0.2, x0 + 0.2, color="#e5487a" if e < 0 else "#0f9d8a", lw=3)
        if e < 0:
            ax.annotate("↑↓", (x0, e), ha="center", va="bottom", fontsize=12)
ax.set_xlim(-1, 1); ax.set_xticks([]); ax.set_ylabel("E − α  (units of |β|)")
ax.set_title("Benzene π MOs: 6 electrons fill the bonding set")
plt.show()
```

Chapter 28 extends this to any conjugated molecule.

## Random numbers and simulation

**Random walk diffusion** of 2 000 molecules:

```{code-cell} ipython3
rng = np.random.default_rng(1)
steps = rng.choice([-1, 1], size=(2000, 500, 2))
paths = steps.cumsum(axis=1)
msd = (paths ** 2).sum(axis=2).mean(axis=0)

fig, axes = plt.subplots(1, 2, figsize=(9, 3.8))
for p in paths[:8]:
    axes[0].plot(*p.T, lw=0.8)
axes[0].set_title("8 random walkers"); axes[0].set_aspect("equal")
axes[1].plot(msd, color="#6c4ad6"); axes[1].set_xlabel("step"); axes[1].set_ylabel("MSD")
axes[1].set_title("Mean-square displacement grows linearly")
plt.tight_layout(); plt.show()
```

## Exercises

**★ 1.** Create an array of the first 20 atomic numbers and select the even ones with a mask.

**★★ 2.** Use `np.linalg.svd` to balance $\mathrm{Fe_2O_3 + C \rightarrow Fe + CO_2}$.

**★★★ 3.** Build the Hückel matrix for 1,3-butadiene (a chain, not a ring) and show that the HOMO–LUMO gap is 1.236|β|.
