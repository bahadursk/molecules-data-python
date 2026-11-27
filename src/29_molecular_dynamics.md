---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Molecular Simulation and MD Analysis

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 12, 13 and 16, plus basic statistical thermodynamics.
```

Molecular simulation connects the forces between atoms to the properties we measure: pressure, diffusion, structure, free energy. A beaker of liquid contains around 10²³ molecules, far too many to track, but the bulk properties of a liquid emerge from a few hundred or thousand molecules if we model them carefully. **Statistical mechanics** provides the link: a macroscopic property is an *average* over many microscopic arrangements of the atoms. Simulations generate those arrangements in one of two ways. **Monte Carlo** (MC) produces random configurations with the correct Boltzmann probabilities; **molecular dynamics** (MD) follows the atoms through time by solving Newton's equations of motion. MD therefore also gives dynamic properties such as diffusion. We will write a Monte Carlo code and a molecular dynamics (MD) engine from scratch in NumPy, then analyse the trajectories just as research codes do.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(2026)
```

Throughout we use **reduced Lennard-Jones units**: lengths in σ, energies in ε, so that argon has σ = 3.40 Å and ε/k_B = 120 K. In these units the numbers stay close to 1, and one simulation describes every substance that follows the same model, simply rescaled.

## The Lennard-Jones fluid

The atoms interact through the **Lennard-Jones potential** (Chapter 16), $U(r) = 4\varepsilon[(\sigma/r)^{12} - (\sigma/r)^6]$, which in reduced units becomes $4(r^{-12} - r^{-6})$. Its minimum lies at $r = 2^{1/6} \approx 1.12$ with depth −1:

```{code-cell} ipython3
r_demo = np.array([1.0, 2 ** (1 / 6), 1.5, 2.5])
print((4 * (r_demo**-12 - r_demo**-6)).round(3))
```

Three techniques make a small box behave like bulk liquid. **Periodic boundary conditions** surround the box with copies of itself, so a particle leaving through one face re-enters through the opposite one and there are no walls. The **minimum-image convention** measures each distance to the nearest copy of the other particle, which is what `d -= L * np.round(d / L)` does. A **cut-off** ignores interactions beyond $r_c = 2.5\sigma$, where the potential is tiny, and the energy is shifted so that it goes smoothly to zero there. The function below computes all pair distances at once with NumPy broadcasting (Chapter 12), then the total energy, the force on every particle (the negative gradient of the energy) and the *virial*, which is needed for the pressure. A second helper places the particles on a cubic lattice as a tidy starting point.

```{code-cell} ipython3
def lj_energy_forces(pos, L, rc=2.5):
    """Total LJ energy and forces with the minimum-image convention."""
    d = pos[:, None, :] - pos[None, :, :]
    d -= L * np.round(d / L)
    r2 = (d**2).sum(-1)
    np.fill_diagonal(r2, np.inf)
    mask = r2 < rc**2
    inv6 = np.where(mask, 1 / r2**3, 0)
    e_shift = 4 * (rc**-12 - rc**-6)
    E = 0.5 * np.sum(np.where(mask, 4 * (inv6**2 - inv6) - e_shift, 0))
    fmag = np.where(mask, 24 * (2 * inv6**2 - inv6) / r2, 0)
    F = (fmag[:, :, None] * d).sum(axis=1)
    virial = 0.5 * np.sum(fmag * np.where(mask, r2, 0))
    return E, F, virial

def lattice(n_side, L):
    g = (np.arange(n_side) + 0.5) * L / n_side
    return np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
```

## Metropolis Monte Carlo

At temperature $T$, the probability of a configuration with energy $U$ is proportional to the Boltzmann factor $e^{-U/k_BT}$. The **Metropolis algorithm** generates configurations with exactly these probabilities without ever computing the (impossibly large) sum over all configurations. At each step it proposes a small random move of one particle and computes the energy change ΔU. A move that lowers the energy is always accepted; a move that raises it is accepted with probability $e^{-\Delta U/k_BT}$, by comparing that number with a uniform random number between 0 and 1. Uphill moves are therefore possible but rare, which is how the simulation explores thermal fluctuations rather than simply sliding into the lowest-energy structure.

The acceptance rule is the heart of the method and is easy to test on its own:

```{code-cell} ipython3
def accept(dE, T, rng):
    return dE < 0 or rng.random() < np.exp(-dE / T)

trials = [accept(1.0, 1.0, rng) for _ in range(10_000)]
print(f"uphill move of ΔU = 1 at T = 1 accepted {np.mean(trials):.1%} of the time (theory {np.exp(-1):.1%})")
```

Move one particle at random; accept with probability $\min(1, e^{-\Delta U / k_BT})$. Only the moved particle's interactions change, so the energy difference is computed from that particle alone, which keeps each step cheap. The maximum step size `dmax` is chosen so that a reasonable fraction of moves (roughly 30–50 %) is accepted.

```{code-cell} ipython3
def mc_run(N_side=5, rho=0.8, T=1.2, steps=6000, dmax=0.12):
    N = N_side**3
    L = (N / rho) ** (1 / 3)
    pos = lattice(N_side, L)
    E, _, _ = lj_energy_forces(pos, L)
    energies, accepted = [], 0
    for step in range(steps):
        i = rng.integers(N)
        old = pos[i].copy()
        def ei(p):
            d = p - np.delete(pos, i, axis=0); d -= L * np.round(d / L)
            r2 = (d**2).sum(-1); r2 = r2[r2 < 6.25]
            return np.sum(4 * (r2**-6 - r2**-3))
        e_old = ei(old)
        pos[i] = (old + rng.uniform(-dmax, dmax, 3)) % L
        dE = ei(pos[i]) - e_old
        if dE < 0 or rng.random() < np.exp(-dE / T):
            E += dE; accepted += 1
        else:
            pos[i] = old
        energies.append(E / N)
    return np.array(energies), accepted / steps, pos, L

E_mc, acc, pos_mc, L_mc = mc_run()
plt.plot(E_mc, color="#6c4ad6"); plt.xlabel("MC step"); plt.ylabel("U/N (ε)")
plt.title(f"Equilibration of 125 LJ particles (acceptance {acc:.0%})"); plt.show()
```

The energy per particle starts high (the lattice is an unnatural arrangement for a liquid) and settles to a fluctuating plateau. That initial stretch is **equilibration** and is discarded before averaging.

## Molecular dynamics with velocity Verlet

Molecular dynamics integrates Newton's second law, $\mathbf{F} = m\mathbf{a}$, for every atom: the forces from the potential give the accelerations, which update the velocities, which update the positions, in small time steps $\Delta t$. The standard integrator is **velocity Verlet**, which updates positions with the current acceleration and velocities with the average of the old and new accelerations:

$$\mathbf r(t+\Delta t) = \mathbf r + \mathbf v\Delta t + \tfrac12\mathbf a\Delta t^2,\qquad \mathbf v(t+\Delta t) = \mathbf v + \tfrac12[\mathbf a(t) + \mathbf a(t+\Delta t)]\Delta t$$

It is preferred over simpler schemes because it is *time-reversible* and conserves the total energy very well over long runs. A single particle on a spring (a harmonic oscillator with force $F = -x$) shows the idea; the energy stays almost exactly constant:

```{code-cell} ipython3
x_o, v_o, dt_o = 1.0, 0.0, 0.05
energies_o = []
for _ in range(2000):
    v_o += 0.5 * (-x_o) * dt_o           # half-step velocity
    x_o += v_o * dt_o                    # full-step position
    v_o += 0.5 * (-x_o) * dt_o           # second half-step with the new force
    energies_o.append(0.5 * v_o**2 + 0.5 * x_o**2)
print(f"energy drift over 2000 steps: {max(energies_o) - min(energies_o):.1e}")
```

The full engine applies the same three lines to all 125 atoms. Initial velocities are drawn from a Gaussian with variance $k_BT$, the Maxwell–Boltzmann distribution, and their average is removed so the box does not drift. The **temperature** is computed from the kinetic energy through equipartition, $\frac{3}{2}Nk_BT = \text{KE}$ (with $3N - 3$ degrees of freedom after removing the drift). During equilibration a crude **thermostat** rescales the velocities to the target temperature at every step; afterwards it is switched off and the system evolves freely at constant energy (the *microcanonical* or NVE ensemble). The pressure combines the ideal-gas term with the virial from the forces. `unwrapped` keeps positions without folding them back into the box, which is needed later to measure how far atoms travel.

```{code-cell} ipython3
def md_run(N_side=5, rho=0.8, T0=1.0, dt=0.004, steps=2500, thermostat_until=800):
    N = N_side**3
    L = (N / rho) ** (1 / 3)
    pos = lattice(N_side, L) + rng.normal(0, 0.02, (N, 3))
    vel = rng.normal(0, np.sqrt(T0), (N, 3)); vel -= vel.mean(0)
    E, F, W = lj_energy_forces(pos, L)
    unwrapped = pos.copy()
    traj, log = [], []
    for step in range(steps):
        vel += 0.5 * F * dt
        new = pos + vel * dt
        unwrapped += vel * dt
        pos = new % L
        E, F, W = lj_energy_forces(pos, L)
        vel += 0.5 * F * dt
        KE = 0.5 * np.sum(vel**2)
        T = 2 * KE / (3 * N - 3)
        if step < thermostat_until:          # simple velocity rescaling during equilibration
            vel *= np.sqrt(T0 / T)
        P = rho * T + W / (3 * L**3)
        log.append((step * dt, E / N, KE / N, (E + KE) / N, T, P))
        if step % 10 == 0:
            traj.append(unwrapped.copy())
    return np.array(log), np.array(traj), pos, L

log, traj, pos, L = md_run()
fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
t = log[:, 0]
axes[0].plot(t, log[:, 1], color="#e5487a", label="potential"); axes[0].plot(t, log[:, 2], color="#0f9d8a", label="kinetic")
axes[0].plot(t, log[:, 3], color="k", label="total"); axes[0].axvline(800 * 0.004, ls=":", color="grey")
axes[0].set(xlabel="t (τ)", ylabel="E/N (ε)", title="Energy conservation after thermostat off"); axes[0].legend(fontsize=8)
axes[1].plot(t, log[:, 4], color="#f2a93b"); axes[1].set(xlabel="t (τ)", title="Temperature")
axes[2].plot(t, log[:, 5], color="#6c4ad6", alpha=0.7); axes[2].set(xlabel="t (τ)", title="Pressure")
plt.tight_layout(); plt.show()
```

After the thermostat is switched off (dotted line), the total energy stays flat while kinetic and potential energy exchange, which is the sign of a correct integrator and a small enough time step. The temperature then fluctuates around its target.

## Radial distribution function

$g(r)$ is the probability of finding a neighbour at distance *r*, relative to an ideal gas. It is measured directly by X-ray and neutron scattering. In an ideal gas, particles are scattered at random and $g(r) = 1$ everywhere. In a liquid, atoms cannot overlap, so $g(r) = 0$ at short range, and they pack into shells of neighbours, which give peaks at about one, two and three molecular diameters that die away at long range, where the liquid looks disordered. A crystal would instead show sharp peaks at fixed lattice distances. To compute $g(r)$, collect all pair distances in each saved frame into a histogram, then divide each bin by the number of pairs an ideal gas of the same density would put in that spherical shell, $4\pi r^2\,\Delta r\,\rho$.

```{code-cell} ipython3
def rdf(frames, L, nbins=80, rmax=None):
    rmax = rmax or L / 2
    hist = np.zeros(nbins)
    for f in frames:
        x = f % L
        d = x[:, None, :] - x[None, :, :]; d -= L * np.round(d / L)
        r = np.sqrt((d**2).sum(-1))[np.triu_indices(len(x), 1)]
        h, edges = np.histogram(r, bins=nbins, range=(0, rmax))
        hist += h
    N = frames.shape[1]; rho = N / L**3
    rc = 0.5 * (edges[1:] + edges[:-1])
    shell = 4 * np.pi * rc**2 * (edges[1] - edges[0])
    return rc, hist / (len(frames) * N / 2 * rho * shell)

r, g = rdf(traj[100:], L)
plt.plot(r, g, color="#6c4ad6", lw=2.5); plt.axhline(1, color="grey", ls=":")
plt.fill_between(r, g, 1, where=g > 1, color="#6c4ad6", alpha=0.15)
plt.xlabel("r (σ)"); plt.ylabel("g(r)"); plt.title("Liquid structure: neighbour shells"); plt.show()
```

## Diffusion from the mean-square displacement

In a liquid each atom wanders by a random walk as it is jostled by its neighbours. Its average squared distance from its starting point, the **mean-square displacement** (MSD), grows linearly with time, and the slope gives the **self-diffusion coefficient** $D$ through the Einstein relation:

$$\langle |\mathbf r(t) - \mathbf r(0)|^2\rangle = 6Dt$$

The factor 6 comes from three dimensions, each contributing $2Dt$. A random walk generated with NumPy confirms the relation before we apply it to the simulation:

```{code-cell} ipython3
steps_rw = rng.normal(0, 1, (5000, 200, 3))          # 200 walkers, unit-variance steps
paths = np.cumsum(steps_rw, axis=0)
msd_rw = (paths**2).sum(-1).mean(1)
print(f"MSD after 5000 steps: {msd_rw[-1]:.0f}  (theory 3 × 5000 = 15000)")
```

For the MD trajectory, the MSD is averaged over all atoms and over every possible starting frame (each time *lag* uses all pairs of frames that far apart), which reduces noise. The first few lags are skipped in the fit because at very short times atoms move ballistically rather than diffusively. Converting the reduced result to SI units with argon's σ and time unit τ gives a value of the same order as the measured diffusion coefficient of liquid argon, a few times 10⁻⁹ m² s⁻¹.

```{code-cell} ipython3
prod = traj[100:]
lags = np.arange(1, 80)
msd = np.array([np.mean(np.sum((prod[l:] - prod[:-l])**2, axis=-1)) for l in lags])
dt_frame = 10 * 0.004
D = np.polyfit(lags[20:] * dt_frame, msd[20:], 1)[0] / 6
plt.plot(lags * dt_frame, msd, "o", ms=3, color="#0f9d8a")
plt.plot(lags * dt_frame, 6 * D * lags * dt_frame, color="k", ls="--", label=f"D = {D:.3f} σ²/τ")
plt.xlabel("t (τ)"); plt.ylabel("MSD (σ²)"); plt.legend(); plt.title("Self-diffusion"); plt.show()
sigma, tau = 3.40e-10, 2.16e-12          # argon: σ in m, τ in s
print(f"D(argon, reduced → SI) ≈ {D * sigma**2 / tau * 1e9:.2f} × 10⁻⁹ m²/s")
```

## Visualising a snapshot

A simulation snapshot is just coordinates, so it can be written in XYZ format (Chapter 8), here converted to ångströms for argon, and shown with py3Dmol (Chapter 23). Colouring by height makes the 3D arrangement easier to see.

```{code-cell} ipython3
import py3Dmol
xyz = f"{len(pos)}\nLJ snapshot\n" + "\n".join(f"Ar {x*3.4:.3f} {y*3.4:.3f} {z*3.4:.3f}" for x, y, z in pos)
v = py3Dmol.view(width=420, height=360)
v.addModel(xyz, "xyz"); v.setStyle({"sphere": {"scale": 0.45, "colorscheme": {"prop": "z", "gradient": "roygb",
                                                                              "min": 0, "max": L * 3.4}}})
v.zoomTo(); v.show()
```

## Analysing research trajectories with MDAnalysis

Our engine handles a hundred atoms; research simulations of proteins in water contain hundreds of thousands, use detailed force fields with bonds, angles, torsions and electrostatics, and run for microseconds on GPUs. Production simulations are run with GROMACS, OpenMM, LAMMPS or AMBER. Each writes a **topology** file (which atoms, residues and bonds exist) and a **trajectory** file (coordinates frame by frame). The analysis ideas are the same as above: RMSD measures how far a structure drifts from a reference, and $g(r)$ describes solvation shells. **MDAnalysis** {cite:p}`michaud2011` reads all their formats. This code needs your own topology and trajectory files, so it is shown but not run here:

```python
import MDAnalysis as mda
from MDAnalysis.analysis import rms, rdf as mda_rdf

u = mda.Universe("protein.pdb", "trajectory.xtc")
backbone = u.select_atoms("backbone")
R = rms.RMSD(u, select="backbone").run()
print(R.results.rmsd[:5])                     # frame, time, RMSD (Å)

water_O = u.select_atoms("resname SOL and name OW")
g = mda_rdf.InterRDF(water_O, water_O, range=(0, 10)).run()
```

```{admonition} Summary
:class: tip
- Simulations average microscopic configurations to predict bulk properties; periodic boundaries, minimum image and a cut-off make a small box act like bulk.
- Metropolis MC accepts downhill moves always and uphill moves with probability $e^{-\Delta U/k_BT}$.
- MD integrates Newton's equations with velocity Verlet, which conserves energy; temperature comes from kinetic energy.
- Discard equilibration before averaging.
- $g(r)$ reveals liquid structure; the MSD slope gives the diffusion coefficient, $\text{MSD} = 6Dt$.
- MDAnalysis applies the same analyses to trajectories from research codes.
```

## Exercises

**★ 1.** Run the MD at ρ = 0.5 and ρ = 1.0. How do g(r) and D change?

**★★ 2.** Compute the heat capacity from the fluctuations of the total energy in an NVT Monte Carlo run: $C_V = \langle\Delta U^2\rangle / k_B T^2$.

**★★★ 3.** Add a Berendsen thermostat and compare the kinetic-energy distribution with the Maxwell–Boltzmann prediction.
