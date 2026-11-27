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

Molecular simulation connects the forces between atoms to the properties we measure: pressure, diffusion, structure, free energy. We will write a Monte Carlo code and a molecular dynamics (MD) engine from scratch in NumPy, then analyse the trajectories just as research codes do.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(2026)
```

Throughout we use **reduced Lennard-Jones units**: lengths in σ, energies in ε, so that argon has σ = 3.40 Å and ε/k_B = 120 K.

## The Lennard-Jones fluid

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
    virial = 0.5 * np.sum(fmag * r2)
    return E, F, virial

def lattice(n_side, L):
    g = (np.arange(n_side) + 0.5) * L / n_side
    return np.array(np.meshgrid(g, g, g)).reshape(3, -1).T
```

## Metropolis Monte Carlo

Move one particle at random; accept with probability $\min(1, e^{-\Delta U / k_BT})$.

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

## Molecular dynamics with velocity Verlet

$$\mathbf r(t+\Delta t) = \mathbf r + \mathbf v\Delta t + \tfrac12\mathbf a\Delta t^2,\qquad \mathbf v(t+\Delta t) = \mathbf v + \tfrac12[\mathbf a(t) + \mathbf a(t+\Delta t)]\Delta t$$

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

## Radial distribution function

$g(r)$ is the probability of finding a neighbour at distance *r*, relative to an ideal gas. It is measured directly by X-ray and neutron scattering.

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

$$\langle |\mathbf r(t) - \mathbf r(0)|^2\rangle = 6Dt$$

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

```{code-cell} ipython3
import py3Dmol
xyz = f"{len(pos)}\nLJ snapshot\n" + "\n".join(f"Ar {x*3.4:.3f} {y*3.4:.3f} {z*3.4:.3f}" for x, y, z in pos)
v = py3Dmol.view(width=420, height=360)
v.addModel(xyz, "xyz"); v.setStyle({"sphere": {"scale": 0.45, "colorscheme": {"prop": "z", "gradient": "roygb",
                                                                              "min": 0, "max": L * 3.4}}})
v.zoomTo(); v.show()
```

## Analysing research trajectories with MDAnalysis

Production simulations are run with GROMACS, OpenMM, LAMMPS or AMBER. **MDAnalysis** {cite:p}`michaud2011` reads all their formats. This code needs your own topology and trajectory files, so it is shown but not run here:

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

## Exercises

**★ 1.** Run the MD at ρ = 0.5 and ρ = 1.0. How do g(r) and D change?

**★★ 2.** Compute the heat capacity from the fluctuations of the total energy in an NVT Monte Carlo run: $C_V = \langle\Delta U^2\rangle / k_B T^2$.

**★★★ 3.** Add a Berendsen thermostat and compare the kinetic-energy distribution with the Maxwell–Boltzmann prediction.
