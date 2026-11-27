---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Spectroscopy and Signal Processing

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 12, 13, 16 and 20.
```

Spectra are how chemists see molecules. This chapter processes UV–Vis, IR, NMR and mass spectra in Python: calibration, peak picking, baseline correction, multiplet simulation, Fourier transforms and isotope patterns.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize, sparse
from scipy.sparse.linalg import spsolve
rng = np.random.default_rng(11)
```

## UV–Vis: Beer–Lambert calibration

$$A = \varepsilon\,l\,c$$

```{code-cell} ipython3
wl = np.linspace(350, 700, 351)
band = np.exp(-0.5 * ((wl - 525) / 35) ** 2)            # KMnO4-like band
concs = np.array([2, 4, 6, 8, 10]) * 1e-5                 # mol/L
eps = 2400 / 1e-5 / 100                                    # L/(mol cm) at the peak, illustrative
spectra = np.array([eps * c * band + rng.normal(0, 0.004, wl.size) for c in concs])

fig, axes = plt.subplots(1, 2, figsize=(11, 4))
for s, c, col in zip(spectra, concs, plt.cm.RdPu(np.linspace(0.35, 1, 5))):
    axes[0].plot(wl, s, color=col, lw=2, label=f"{c*1e5:.0f}×10⁻⁵ M")
axes[0].set(xlabel="λ / nm", ylabel="A", title="Calibration standards"); axes[0].legend(fontsize=8)

A525 = spectra[:, np.argmin(abs(wl - 525))]
slope, intercept = np.polyfit(concs, A525, 1)
unknown_A = 0.61
c_unknown = (unknown_A - intercept) / slope
axes[1].plot(concs * 1e5, A525, "o", color="#6c4ad6", ms=9)
axes[1].plot(concs * 1e5, np.polyval([slope, intercept], concs), color="#6c4ad6")
axes[1].axhline(unknown_A, ls="--", color="#e5487a"); axes[1].axvline(c_unknown * 1e5, ls="--", color="#e5487a")
axes[1].set(xlabel="c / 10⁻⁵ M", ylabel="A at 525 nm", title=f"Unknown: {c_unknown*1e5:.2f}×10⁻⁵ M")
plt.tight_layout(); plt.show()
```

### Spectrum to perceived colour

```{code-cell} ipython3
def wavelength_to_rgb(l):
    """Approximate RGB for a visible wavelength (nm)."""
    if l < 440: r, g, b = -(l - 440) / 60, 0, 1
    elif l < 490: r, g, b = 0, (l - 440) / 50, 1
    elif l < 510: r, g, b = 0, 1, -(l - 510) / 20
    elif l < 580: r, g, b = (l - 510) / 70, 1, 0
    elif l < 645: r, g, b = 1, -(l - 645) / 65, 0
    else: r, g, b = 1, 0, 0
    return np.clip([r, g, b], 0, 1)

def perceived_colour(wl, A):
    T = 10 ** -A
    rgb = sum(wavelength_to_rgb(l) * t for l, t in zip(wl, T) if 380 <= l <= 700)
    return rgb / rgb.max()

fig, ax = plt.subplots(figsize=(9, 1.6))
for i, (s, c) in enumerate(zip(spectra, concs)):
    ax.add_patch(plt.Rectangle((i, 0), 0.9, 1, color=perceived_colour(wl, s)))
    ax.text(i + 0.45, -0.3, f"{c*1e5:.0f}×10⁻⁵ M", ha="center", fontsize=8)
ax.set_xlim(0, 5); ax.set_ylim(-0.5, 1); ax.axis("off"); ax.set_title("Colour of the solutions computed from their spectra")
plt.show()
```

## IR: baseline correction and peak picking

Real spectra sit on curved baselines. **Asymmetric least squares** (ALS) fits a smooth curve that hugs the bottom of the spectrum.

```{code-cell} ipython3
wn = np.linspace(4000, 500, 3500)
bands = [(3400, 150, 0.45), (2960, 25, 0.35), (2870, 20, 0.2), (1735, 18, 0.95), (1460, 20, 0.3),
         (1370, 15, 0.35), (1240, 30, 0.7), (1045, 25, 0.5)]
true = sum(h * np.exp(-0.5 * ((wn - c) / w) ** 2) for c, w, h in bands)
baseline = 0.15 + 0.1 * np.sin(wn / 900) + 2e-5 * (4000 - wn)
measured = true + baseline + rng.normal(0, 0.006, wn.size)

def als_baseline(y, lam=1e6, p=0.01, n_iter=10):
    L = y.size
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    D = lam * D.dot(D.T)
    w = np.ones(L)
    for _ in range(n_iter):
        W = sparse.spdiags(w, 0, L, L)
        z = spsolve(W + D, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

bl = als_baseline(measured)
corrected = measured - bl
peaks, _ = signal.find_peaks(corrected, prominence=0.1)

fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
axes[0].plot(wn, measured, color="grey", label="measured"); axes[0].plot(wn, bl, color="#e5487a", lw=2, label="ALS baseline")
axes[0].legend(); axes[0].set_ylabel("A")
axes[1].plot(wn, corrected, color="#6c4ad6")
assign = {3400: "O–H", 2960: "C–H", 1735: "C=O ester", 1240: "C–O", 1045: "C–O", 1370: "CH₃", 1460: "CH₂"}
for p in peaks:
    lab = min(assign, key=lambda k: abs(k - wn[p]))
    axes[1].annotate(f"{wn[p]:.0f}\n{assign[lab]}", (wn[p], corrected[p]), xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=8, color="#0f9d8a")
axes[1].set(xlabel="Wavenumber / cm$^{-1}$", ylabel="A (corrected)"); axes[1].invert_xaxis()
plt.tight_layout(); plt.show()
```

## NMR: from multiplets to the Fourier transform

### Simulating first-order multiplets

A proton coupled to *n* equivalent neighbours splits into *n* + 1 lines with intensities from Pascal's triangle.

```{code-cell} ipython3
from math import comb

def multiplet(center, n_neighbours, J_Hz, n_H, field_MHz=400):
    J = J_Hz / field_MHz
    lines = [(center + (k - n_neighbours / 2) * J, n_H * comb(n_neighbours, k) / 2 ** n_neighbours)
             for k in range(n_neighbours + 1)]
    return lines

def spectrum(lines, x, width=0.0015):
    return sum(h * width**2 / ((x - c) ** 2 + width**2) for c, h in lines)

x = np.linspace(0, 5, 20000)
# 1-propanol CH3CH2CH2OH
signals = {"CH₃ (t)": multiplet(0.93, 2, 7.4, 3), "CH₂ (sextet)": multiplet(1.57, 5, 7.0, 2),
           "OCH₂ (t)": multiplet(3.58, 2, 6.7, 2), "OH (s)": multiplet(2.3, 0, 0, 1)}
fig, ax = plt.subplots(figsize=(10, 3.8))
for (name, lines), c in zip(signals.items(), ["#e5487a", "#f2a93b", "#0f9d8a", "#6c4ad6"]):
    y = spectrum(lines, x)
    ax.plot(x, y, color=c, lw=1.5)
    ax.text(lines[len(lines) // 2][0], y.max() * 1.05, name, ha="center", color=c, fontsize=9)
ax.invert_xaxis(); ax.set(xlabel="δ / ppm", yticks=[], title="Simulated ¹H NMR of 1-propanol (400 MHz)")
plt.show()
```

### The free induction decay and FFT

An NMR spectrometer actually records a decaying oscillation in time, the **FID**. A Fourier transform turns it into the spectrum.

```{code-cell} ipython3
sw, N = 4000.0, 8192                       # spectral width (Hz), points
t = np.arange(N) / sw
freqs_Hz = [400, 420, 440, 1250, 1270, 2100]   # offsets of individual lines
amps = [1, 2, 1, 1, 1, 0.5]
T2 = 0.4
fid = sum(a * np.exp(2j * np.pi * f * t) for f, a in zip(freqs_Hz, amps)) * np.exp(-t / T2)
fid += (rng.normal(0, 0.05, N) + 1j * rng.normal(0, 0.05, N))

spec = np.fft.fftshift(np.fft.fft(fid * np.exp(-t * np.pi * 1.0)))   # 1 Hz exponential line broadening
f_axis = np.fft.fftshift(np.fft.fftfreq(N, 1 / sw))

fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
axes[0].plot(t[:1500], fid.real[:1500], color="#0f9d8a", lw=0.7); axes[0].set(xlabel="t / s", title="FID (real part)")
axes[1].plot(f_axis, spec.real, color="#6c4ad6"); axes[1].set(xlim=(2300, 200), xlabel="frequency offset / Hz", title="Spectrum after FFT")
plt.tight_layout(); plt.show()
```

### Predicting shifts with RDKit-based rules

A crude but instructive predictor assigns each hydrogen-bearing carbon a base shift by its environment.

```{code-cell} ipython3
from rdkit import Chem
rules = [("[CH3][CX4]", 0.9, "alkyl CH₃"), ("[CH2X4]([CX4])[CX4]", 1.3, "alkyl CH₂"), ("[CX4;H1,H2,H3][CX3]=O", 2.1, "α to C=O"),
         ("[CX4;H1,H2,H3]c", 2.3, "benzylic"), ("[CX4;H1,H2,H3][OX2]", 3.6, "next to O"), ("[cH]", 7.2, "aromatic"),
         ("[CX3H1]=O", 9.7, "aldehyde"), ("[OX2H][CX3]=O", 11.5, "carboxylic acid")]
def predict_shifts(smiles):
    m = Chem.MolFromSmiles(smiles)
    out = {}
    for smarts, shift, label in rules:
        for match in m.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
            out.setdefault(match[0], (shift, label))
    return out

for smi in ["CCOC(C)=O", "Cc1ccccc1", "CC(=O)O"]:
    print(smi, sorted({v for v in predict_shifts(smi).values()}))
```

Chapter 39 replaces such hand-written rules with a machine-learned shift predictor.

## Mass spectrometry: isotope patterns

The isotope pattern of a molecule is the convolution of its elements' isotope distributions.

```{code-cell} ipython3
isotopes = {"C": [(12.000, 0.9893), (13.003, 0.0107)], "H": [(1.008, 0.99988), (2.014, 0.00012)],
            "Cl": [(34.969, 0.7576), (36.966, 0.2424)], "Br": [(78.918, 0.5069), (80.916, 0.4931)],
            "O": [(15.995, 0.99757), (16.999, 0.00038), (17.999, 0.00205)], "N": [(14.003, 0.99636), (15.000, 0.00364)]}

def isotope_pattern(formula):
    dist = {0: 1.0}                      # nominal-mass offsets
    mono = 0
    for el, n in formula.items():
        base = isotopes[el][0][0]
        mono += n * base
        for _ in range(n):
            new = {}
            for off, p in dist.items():
                for mass, ab in isotopes[el]:
                    k = off + round(mass - base)
                    new[k] = new.get(k, 0) + p * ab
            dist = {k: v for k, v in new.items() if v > 1e-6}
    m = max(dist.values())
    return {round(mono) + k: 100 * v / m for k, v in sorted(dist.items())}

fig, axes = plt.subplots(1, 3, figsize=(12, 3.4))
for ax, (name, f), c in zip(axes, [("C₆H₅Cl", {"C": 6, "H": 5, "Cl": 1}), ("CH₂Br₂", {"C": 1, "H": 2, "Br": 2}),
                                    ("C₆₀", {"C": 60})], ["#0f9d8a", "#e5487a", "#6c4ad6"]):
    pat = isotope_pattern(f)
    ax.bar(list(pat), list(pat.values()), color=c, width=0.4)
    ax.set(title=name, xlabel="m/z"); ax.set_xticks(list(pat)[:6])
axes[0].set_ylabel("relative intensity / %"); plt.tight_layout(); plt.show()
```

Chlorine gives the familiar 3 : 1 M : M+2 pattern, two bromines a 1 : 2 : 1 triplet, and C₆₀ has so many carbons that M+1 is two-thirds the height of M.

## Exercises

**★ 1.** Use the calibration line to find the concentrations of solutions with A = 0.25, 0.48 and 0.95.

**★★ 2.** Simulate the ¹H NMR spectrum of ethyl acetate at 60 MHz and at 600 MHz. What changes, and why?

**★★★ 3.** Fit three Lorentzians to the overlapping C–H stretches of the corrected IR spectrum and report their areas.
