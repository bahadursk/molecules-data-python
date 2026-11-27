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

Almost every number a chemist records comes in bulk. A UV–vis spectrum is a few thousand absorbance values, one for each wavelength. A molecular dynamics trajectory is millions of atomic coordinates. A microscope image is a grid of brightness values, and a titration is a column of volumes beside a column of pH readings. In Part I we stored such data in Python lists, which works, but lists were designed to hold *anything* (a string next to a number next to another list), and that flexibility makes them slow and awkward for arithmetic. If you want to convert a list of 4 000 temperatures from °C to K, you have to write a loop.

**NumPy** (Numerical Python) solves this with one new object, the **array**: a block of numbers that all share one type and sit side by side in memory. Because NumPy knows every element is, say, a 64-bit decimal number, it can hand the whole block to fast, pre-compiled code and do the arithmetic for every element in one step. That is why NumPy is the foundation of the whole scientific Python world: pandas tables, Matplotlib plots, SciPy fitting, RDKit fingerprints and scikit-learn models all store their numbers in NumPy arrays underneath. Learning arrays well pays off in every later chapter.

By convention NumPy is always imported under the short name `np`, so that you write `np.sqrt` rather than `numpy.sqrt`. You will see this line at the top of almost every scientific notebook in the world.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)   # print 4 decimals, no 1e-05 style for small numbers
```

## Creating arrays

The simplest way to make an array is to hand `np.array` a list you already have. Here are the atomic masses of H, C, N and O. Compare how a list and an array respond to the same multiplication:

```{code-cell} ipython3
masses_list = [1.008, 12.011, 14.007, 15.999]
masses = np.array(masses_list)

print(masses_list * 2)    # a list repeats itself
print(masses * 2)         # an array doubles every number
```

The list simply joined two copies of itself, because for lists `*` means "repeat". The array did what a chemist means: it multiplied each mass by two. This one difference (arithmetic on arrays acts on the *numbers*) is the reason arrays exist.

Every array carries three pieces of information you will check constantly. Its **dtype** is the type shared by all elements (`float64` means 64-bit decimal numbers, `int64` whole numbers). Its **shape** gives the number of elements along each direction, and **ndim** says how many directions there are.

```{code-cell} ipython3
print(masses.dtype, masses.shape, masses.ndim)
```

`(4,)` means one direction holding four numbers. The trailing comma is how Python writes a tuple with a single entry.

You rarely type arrays by hand. More often you ask NumPy to generate them, and three functions cover most needs. `np.zeros` and `np.ones` create arrays to be filled in later, for example an empty results table before a loop. `np.arange` works like `range` but allows decimal steps; it starts at the first number and stops *before* the second. `np.linspace` is the one to use for plotting and calculations over a range, because you choose how many points you get and both ends are always included.

```{code-cell} ipython3
print(np.zeros(3))
print(np.ones((2, 3)))                          # 2 rows, 3 columns

wavelengths = np.arange(400, 701, 50)           # start, stop (excluded), step
print(wavelengths)

T = np.linspace(273.15, 373.15, 5)              # start, stop (included), number of points
print(T)
```

Notice that `np.arange(400, 701, 50)` needed a stop of 701 to include 700. With decimal steps that rule becomes treacherous, because rounding errors can add or lose a point, which is why `linspace` is usually the safer choice.

## Vectorised arithmetic

When an operation is applied to an array, NumPy applies it to **every element at once**. This is called **vectorisation**, and it replaces most of the loops you wrote in Part I. A temperature conversion that needed a `for` loop over a list becomes a single line, and it reads exactly like the formula on paper:

```{code-cell} ipython3
T_C = np.array([-78.5, 0.0, 25.0, 37.0, 100.0])   # dry ice, ice, room, body, boiling water
T_K = T_C + 273.15
print(T_K)
```

Two arrays of the same shape are combined element by element, so the first number meets the first number, the second meets the second, and so on. For example, number of moles is mass divided by molar mass, worked out for three samples at once:

```{code-cell} ipython3
mass_g = np.array([5.00, 10.0, 2.50])            # grams of NaCl, glucose, water
molar_mass = np.array([58.44, 180.16, 18.015])   # g/mol
moles = mass_g / molar_mass
print(moles)
```

NumPy also provides vectorised versions of the mathematical functions, such as `np.exp`, `np.log`, `np.log10`, `np.sqrt` and `np.sin`. Unlike the `math` module from Chapter 1, they accept whole arrays. Here are Boltzmann factors $e^{-\Delta E / k_B T}$ for an energy gap of $10^{-20}$ J at the five temperatures above. They show how strongly heating populates higher energy levels:

```{code-cell} ipython3
kB = 1.380649e-23                 # J/K
print(np.exp(-1e-20 / (kB * T_K)))
```

pH is another natural example: `-np.log10` turns a whole column of hydronium concentrations into pH values in one step.

```{code-cell} ipython3
H3O = np.array([1e-1, 1e-3, 1e-7, 3.2e-9])
print(-np.log10(H3O))
```

Vectorised code is not only shorter, it is also much faster, because the loop runs in compiled C code instead of the Python interpreter. Here we square and add a million numbers both ways and time them:

```{code-cell} ipython3
import time
x = np.random.default_rng(0).random(1_000_000)
t0 = time.perf_counter(); s1 = sum(v * v for v in x); t1 = time.perf_counter()
s2 = np.sum(x * x); t2 = time.perf_counter()
print(f"Python loop {1000*(t1-t0):.1f} ms, NumPy {1000*(t2-t1):.2f} ms")
```

The exact times depend on your computer, but NumPy is typically one to two hundred times faster. For a single spectrum the difference is unnoticeable. For a screening library of a million molecules, it is the difference between seconds and an afternoon.

## Indexing, slicing and Boolean masks

Getting numbers *out* of an array uses the same square-bracket notation as lists. Counting starts at 0, negative numbers count from the end, and a slice `start:stop` takes everything from `start` up to but not including `stop`:

```{code-cell} ipython3
z = np.arange(1, 11)          # atomic numbers 1 to 10
print(z[0], z[-1])            # first and last
print(z[2:5])                 # elements 3, 4, 5 (Li, Be, B)
print(z[::2])                 # every second element
```

A more powerful kind of indexing has no equivalent for lists. Comparing an array with a number gives an array of `True`/`False` values, one per element, called a **Boolean mask**. Using the mask as an index keeps only the elements where it is `True`. This is how you ask a question of your data ("which values are above the detection limit?") without writing a loop:

```{code-cell} ipython3
readings = np.array([0.002, 0.15, 0.48, 0.004, 0.91])
above_limit = readings > 0.01
print(above_limit)
print(readings[above_limit])
```

Conditions are combined with `&` (and), `|` (or) and `~` (not), and each condition needs its own parentheses. The everyday Python words `and`/`or` do not work on arrays, and forgetting the parentheses is the commonest NumPy error.

Masks are how spectroscopists pick out regions of a spectrum. Below we build a simulated IR spectrum (a sum of Gaussian bands plus noise) on a grid of 3 601 wavenumbers, then use a mask to look only at the carbonyl region between 1650 and 1800 cm⁻¹:

```{code-cell} ipython3
rng = np.random.default_rng(42)
wn = np.linspace(4000, 400, 3601)                  # IR wavenumbers / cm^-1
peaks = [(3350, 120, 0.6), (2950, 40, 0.5), (1715, 25, 0.9), (1250, 40, 0.4)]  # centre, width, height
absorb = sum(h * np.exp(-((wn - c) / w) ** 2) for c, w, h in peaks)
absorb += rng.normal(0, 0.01, wn.size)
transmittance = 10 ** -absorb

print(wn[:3], wn[-3:])
carbonyl = (wn > 1650) & (wn < 1800)             # Boolean mask
print("carbonyl region max A:", absorb[carbonyl].max().round(3),
      "at", wn[carbonyl][absorb[carbonyl].argmax()], "cm^-1")
```

`argmax` returns the *position* of the largest value rather than the value itself, and that position is then used to find the matching wavenumber. The same mask can shade the region on a plot:

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

Many chemical data sets are naturally tables: molecules in rows and properties in columns, or atoms in rows and x, y, z coordinates in columns. A two-dimensional array stores exactly this. You make one from a list of lists, where each inner list becomes a row. You then index it with `[row, column]`, and a colon on its own means "all of them".

```{code-cell} ipython3
grid = np.array([[1, 2, 3],
                 [4, 5, 6]])
print(grid.shape)       # 2 rows, 3 columns
print(grid[1, 2])       # row 1, column 2 -> 6
print(grid[:, 0])       # the whole first column
```

Here are the coordinates of the five atoms of methane, in ångströms, with carbon first. Row 0 is the carbon atom and column 2 holds every z coordinate:

```{code-cell} ipython3
# Coordinates of methane (Å)
xyz = np.array([[ 0.000,  0.000,  0.000],
                [ 0.629,  0.629,  0.629],
                [-0.629, -0.629,  0.629],
                [-0.629,  0.629, -0.629],
                [ 0.629, -0.629, -0.629]])
print(xyz.shape, xyz[0], xyz[:, 2])
```

What happens if you combine arrays of *different* shapes? NumPy follows a rule called **broadcasting**: dimensions of length 1 are stretched to match the other array. Broadcasting is what makes `T_C + 273.15` work, because the single number is stretched to the length of the array. It also lets you subtract one row from every row of a table, for example to move a molecule so that its carbon sits at the origin:

```{code-cell} ipython3
centred = xyz - xyz[0]          # (5, 3) minus (3,) -> the carbon's coordinates removed from every row
print(centred[:2])
```

Taken one step further, broadcasting computes every interatomic distance at once. `xyz[:, None, :]` has shape `(5, 1, 3)` and `xyz[None, :, :]` has shape `(1, 5, 3)`, and subtracting them stretches both to `(5, 5, 3)`: the vector from every atom to every other atom. Taking the length of each vector (`np.linalg.norm` along the last axis) gives the full 5 × 5 distance matrix, with no loop at all. The same matrix underlies molecular mechanics, nearest-neighbour searches and many molecular descriptors in later chapters.

```{code-cell} ipython3
D = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=-1)
print(D.round(3))

v1, v2 = xyz[1] - xyz[0], xyz[2] - xyz[0]
angle = np.degrees(np.arccos(v1 @ v2 / (np.linalg.norm(v1) * np.linalg.norm(v2))))
print(f"H–C–H angle = {angle:.2f}° (tetrahedral 109.47°)")
```

The bond angle used the dot product (`@`): $\cos\theta = \mathbf{v}_1\cdot\mathbf{v}_2 / (|\mathbf{v}_1||\mathbf{v}_2|)$. That is the same geometry you would do by hand, written in one line. A 2-D array can also be drawn directly as a coloured image, which makes patterns in a matrix easy to see: the C–H distances (about 1.09 Å) form the first row and column, and the longer H···H distances fill the rest.

```{code-cell} ipython3
plt.imshow(D, cmap="viridis")
plt.colorbar(label="distance / Å")
plt.xticks(range(5), ["C", "H", "H", "H", "H"]); plt.yticks(range(5), ["C", "H", "H", "H", "H"])
plt.title("Methane distance matrix")
plt.show()
```

## Statistics along axes

Laboratory work is repeated to estimate precision, and NumPy has the summary statistics built in: `mean`, `std`, `min`, `max`, `sum` and `median`. Called on their own they summarise the whole array:

```{code-cell} ipython3
titres = np.array([24.35, 24.40, 24.30, 24.45])   # mL, four titrations
print(titres.mean(), titres.std(ddof=1))
```

`ddof=1` gives the *sample* standard deviation (dividing by $n-1$), which is the one to report for replicate measurements. NumPy's default divides by $n$, and without `ddof=1` you will quietly under-report your uncertainty.

For a table you usually want a statistic for each column or each row, and the `axis` argument chooses which. `axis=0` works *down* the rows and gives one answer per column, while `axis=1` works *across* each row. Here are triplicate absorbance readings (rows) at five concentrations (columns), so `axis=0` averages the three replicates of each standard:

```{code-cell} ipython3
# Triplicate absorbance readings (rows) at five concentrations (columns)
A = np.array([[0.101, 0.198, 0.305, 0.399, 0.502],
              [0.098, 0.203, 0.298, 0.405, 0.497],
              [0.103, 0.201, 0.301, 0.402, 0.505]])
print("mean:", A.mean(axis=0))
print("std: ", A.std(axis=0, ddof=1))
```

## Linear algebra

A surprising amount of chemistry reduces to linear algebra, the mathematics of matrices and vectors. Fitting a calibration line, balancing an equation and finding molecular orbital energies are all, underneath, the same kind of matrix problem. The `np.linalg` module solves them reliably, so you never have to code the algorithms yourself.

### Least-squares calibration line

Beer–Lambert's law says absorbance is proportional to concentration, $A = \varepsilon l c$. To find $\varepsilon l$ from the standards above, we fit the straight line $A = m c + b$ that minimises the sum of squared errors, which is the familiar "line of best fit". `np.linalg.lstsq` does this. We give it a matrix with one column per unknown, the concentrations for the slope and a column of ones for the intercept, and it returns the best slope and intercept.

```{code-cell} ipython3
c = np.array([1, 2, 3, 4, 5]) * 1e-5           # mol/L
y = A.mean(axis=0)
X = np.column_stack([c, np.ones_like(c)])
(slope, intercept), *_ = np.linalg.lstsq(X, y, rcond=None)
print(f"ε·l = {slope:.0f} L/mol, intercept = {intercept:.4f}")
```

An unknown solution with absorbance 0.250 then has concentration $(0.250 - b)/m$. Chapter 16 shows how to get uncertainties on the slope and intercept as well.

### Balancing a chemical equation

Balancing $a\,\mathrm{C_3H_8} + b\,\mathrm{O_2} \rightarrow c\,\mathrm{CO_2} + d\,\mathrm{H_2O}$ means finding coefficients that conserve every element. Each element gives one equation. Carbon, for example, requires $3a - c = 0$. Writing one row per element and one column per species (with products negative) gives a matrix $M$, and the balanced coefficients are the vector that $M$ turns into zero, its so-called *null space*. The singular value decomposition (SVD) finds that vector: it is the last row of `Vt`. We then scale it so the smallest coefficient is 1.

```{code-cell} ipython3
#              C3H8  O2  CO2  H2O
M = np.array([[  3,   0,  -1,   0],   # C
              [  8,   0,   0,  -2],   # H
              [  0,   2,  -2,  -1]])  # O
_, _, Vt = np.linalg.svd(M)
coeffs = Vt[-1] / Vt[-1][np.abs(Vt[-1]) > 1e-9].min()
print(np.round(coeffs, 6))            # 1 C3H8 + 5 O2 -> 3 CO2 + 4 H2O
```

The method needs no guessing and works just as well for reactions with ten species, where balancing by inspection becomes painful.

### Hückel molecular orbitals of benzene

In Hückel theory the energies of the π molecular orbitals are the **eigenvalues** of a matrix with α (the energy of an isolated p orbital) on the diagonal and β (the interaction between bonded neighbours) wherever two carbons are bonded. An eigenvalue problem asks for the special vectors that a matrix only stretches, never rotates. In quantum chemistry those vectors are the orbitals and the stretch factors are their energies. `np.linalg.eigh` solves it for symmetric matrices, returning the energies in increasing order and the orbital coefficients as columns. Using units where α = 0 and β = −1:

```{code-cell} ipython3
n = 6
H = np.zeros((n, n))
for i in range(n):
    H[i, (i + 1) % n] = H[(i + 1) % n, i] = -1.0     # each carbon bonds to the next; % n closes the ring
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

The six π electrons exactly fill the three bonding orbitals (one at −2|β| below α and a degenerate pair at −1|β|), which is the textbook explanation of benzene's aromatic stability. Chapter 28 extends this to any conjugated molecule.

## Random numbers and simulation

Random numbers are not only for games. They let you *simulate* processes governed by chance, such as molecular collisions, diffusion and measurement noise, and they let you test an analysis on data where you already know the right answer. NumPy's modern interface is a **generator** made with `np.random.default_rng(seed)`. Giving a seed makes the "random" sequence repeatable, so your notebook produces the same result every time it runs, which matters for science that others must reproduce.

```{code-cell} ipython3
rng = np.random.default_rng(7)
print(rng.random(3))                  # uniform between 0 and 1
print(rng.normal(25.0, 0.1, 3))       # three simulated thermometer readings: mean 25.0, sd 0.1
print(rng.choice(["H", "C", "N", "O"], 5))
```

A **random walk** is the simplest model of diffusion: at every time step a molecule hops one unit in a random direction. Simulating 2 000 molecules for 500 steps needs no loop. We draw all the ±1 steps at once into a `(2000, 500, 2)` array (molecules × steps × x/y), and `cumsum` along the step axis turns steps into positions. The **mean-square displacement** (MSD), averaged over all molecules, grows in a straight line with time. That linear growth is the signature of diffusion, and its slope gives the diffusion coefficient you will meet again in Chapter 29.

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

```{admonition} Summary
:class: tip
- An array holds numbers of one type side by side, and arithmetic on it acts on every element at once (vectorisation), with no loop.
- Make arrays with `np.array`, `np.zeros`, `np.ones`, `np.arange` and, for ranges, `np.linspace`.
- Select data with indices, slices and Boolean masks combined with `&`, `|` and `~`.
- 2-D arrays are indexed `[row, column]`, and `axis=0` / `axis=1` choose the direction for statistics.
- Broadcasting stretches length-1 dimensions, turning pairwise calculations such as distance matrices into one line.
- `np.linalg` solves least-squares fits, null spaces and eigenvalue problems, and `np.random.default_rng(seed)` gives reproducible simulations.
```

## Exercises

**★ 1.** Create an array of the first 20 atomic numbers and select the even ones with a mask.

**★★ 2.** Use `np.linalg.svd` to balance $\mathrm{Fe_2O_3 + C \rightarrow Fe + CO_2}$.

**★★★ 3.** Build the Hückel matrix for 1,3-butadiene (a chain, not a ring) and show that the HOMO–LUMO gap is 1.236|β|.
