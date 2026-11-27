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

Spectra are how chemists see molecules. A **spectrum** records how a sample interacts with radiation (or, in mass spectrometry, how its ions are sorted by mass) as a function of wavelength, frequency or mass-to-charge ratio. Each technique probes something different: UV–vis absorption shows electronic transitions and measures concentration, infrared (IR) shows bond vibrations and identifies functional groups, nuclear magnetic resonance (NMR) maps the hydrogen and carbon environments, and mass spectrometry (MS) gives the molecular mass and formula. In every case the raw signal must be processed before it can be interpreted: calibrated, cleaned of baseline drift and noise, transformed and assigned. To a computer a spectrum is just two arrays, x and y, so all of this is array arithmetic. This chapter processes UV–Vis, IR, NMR and mass spectra in Python: calibration, peak picking, baseline correction, multiplet simulation, Fourier transforms and isotope patterns.

```{code-cell} ipython3
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, optimize, sparse
from scipy.sparse.linalg import spsolve
rng = np.random.default_rng(11)
```

## UV–Vis: Beer–Lambert calibration

**Absorbance** $A$ measures how much light a solution absorbs: $A = -\log_{10}(I/I_0)$, where $I_0$ and $I$ are the light intensities entering and leaving the sample. The **Beer–Lambert law** says that absorbance is proportional to concentration $c$ and to the path length $l$ of the cell, with the **molar absorptivity** $\varepsilon$ as the constant of proportionality:

$$A = \varepsilon\,l\,c$$

This linear relationship is what makes UV–vis a quantitative technique. In practice $\varepsilon$ is not taken from a table but measured with a **calibration curve**: you record standards of known concentration, fit a straight line of absorbance against concentration, and read the unknown's concentration from the line. The basic arithmetic for one sample is a single division:

```{code-cell} ipython3
A_sample, epsilon_peak, path = 0.42, 6.0e3, 1.0      # absorbance, L/(mol cm), cm
print(f"c = {A_sample / (epsilon_peak * path):.2e} mol/L")
```

Below, five simulated standards of a permanganate-like dye are measured across the visible range. The absorbance at the peak (525 nm) is extracted from each spectrum with `np.argmin(abs(wl - 525))`, the index of the wavelength closest to 525 nm, and fitted with `np.polyfit`. Inverting the line, $c = (A - b)/m$, converts the unknown's absorbance into a concentration.

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

The colour we see is the light a solution *transmits*, so a spectrum predicts a colour. The transmitted fraction at each wavelength is $T = 10^{-A}$. Weighting an approximate RGB colour for each visible wavelength by its transmittance and adding them up gives the perceived colour of the solution, which here deepens from pale pink to purple as the concentration rises. (A rigorous calculation would use the CIE colour-matching functions; this approximation is enough to show the idea.)

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

An IR spectrum shows absorption against **wavenumber** (in cm⁻¹, the number of waves per centimetre, proportional to energy), conventionally plotted from high to low. Characteristic bands identify functional groups: a broad O–H stretch near 3400 cm⁻¹, C–H stretches just below 3000, and a strong C=O stretch near 1700–1750.

Real spectra sit on curved baselines caused by scattering, sample thickness or instrument drift, and a sloping baseline distorts peak heights and areas. **Baseline correction** estimates that slow background and subtracts it. **Asymmetric least squares** (ALS) fits a smooth curve that hugs the bottom of the spectrum. It balances two goals, controlled by two parameters: `lam` sets how smooth the baseline must be (larger is smoother), and `p` sets the asymmetry, so that points above the curve (peaks) are given a tiny weight and points below it a large weight. The fit is repeated a few times, updating the weights, until the curve settles under the peaks. The method is short to write because SciPy's sparse matrices handle the smoothness term efficiently.

```{code-cell} ipython3
wn = np.linspace(4000, 500, 3500)
bands = [(3400, 150, 0.45), (2960, 25, 0.35), (2870, 20, 0.2), (1735, 18, 0.95), (1460, 20, 0.3),
         (1370, 15, 0.35), (1240, 30, 0.7), (1045, 25, 0.5)]
true = sum(h * np.exp(-0.5 * ((wn - c) / w) ** 2) for c, w, h in bands)
baseline = 0.15 + 0.1 * np.sin(wn / 900) + 2e-5 * (4000 - wn)
measured = true + baseline + rng.normal(0, 0.006, wn.size)

def als_baseline(y, lam=1e6, p=0.01, n_iter=10):
    L = y.size
    D = sparse.diags([1.0, -2.0, 1.0], [0, -1, -2], shape=(L, L - 2))
    D = (lam * D.dot(D.T)).tocsc()
    w = np.ones(L)
    for _ in range(n_iter):
        W = sparse.diags(w, 0, format="csc")
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

After correction, `find_peaks` with a prominence threshold picks the bands, and each is labelled with the nearest entry in a small table of assignments. The spectrum is consistent with an ester such as ethyl acetate, with a little water giving the O–H band.

## NMR: from multiplets to the Fourier transform

In ¹H NMR, each chemically distinct hydrogen gives a signal at its own **chemical shift** δ (in ppm), which reflects its electronic environment. Signals are split by **spin–spin coupling** to neighbouring hydrogens, and the spacing between lines is the coupling constant $J$ in hertz. Because $J$ is fixed in Hz while the ppm scale is relative to the spectrometer frequency, a multiplet shrinks on the ppm axis at higher field: 7 Hz is 0.0175 ppm at 400 MHz.

### Simulating first-order multiplets

A proton coupled to *n* equivalent neighbours splits into *n* + 1 lines with intensities from Pascal's triangle. This is the **n + 1 rule**, valid when the shift difference between coupled protons is much larger than $J$ (a *first-order* spectrum). The binomial coefficients $\binom{n}{k}$, available as `math.comb`, give exactly the rows of Pascal's triangle:

```{code-cell} ipython3
from math import comb
for n in range(5):
    print(n, "neighbours:", [comb(n, k) for k in range(n + 1)])
```

The `multiplet` function turns a shift, a number of neighbours and a $J$ into a list of (position, height) lines, and `spectrum` draws each line as a Lorentzian, the natural NMR line shape.

```{code-cell} ipython3
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

An NMR spectrometer does not scan the frequencies one by one. It excites all the nuclei at once with a radio-frequency pulse and records the signal they emit as they relax: a sum of oscillations, one for each line, decaying over time. This decaying signal is the **free induction decay** (FID). The **Fourier transform** converts a signal in time into its component frequencies, turning the FID into the spectrum. The fast algorithm for this, the **FFT**, is `np.fft.fft`. A basic demonstration recovers the frequency of a pure oscillation:

```{code-cell} ipython3
tt = np.arange(256) / 256                     # one second sampled 256 times
wave = np.cos(2 * np.pi * 10 * tt)            # a 10 Hz oscillation
amplitude = np.abs(np.fft.rfft(wave))
print("strongest frequency:", np.argmax(amplitude), "Hz")
```

The simulated FID below contains six lines. The decay time T₂ sets the natural line width (a faster decay gives broader lines). Multiplying the FID by a decaying exponential before the transform, called **line broadening** or apodisation, trades a little resolution for less noise. `fftshift` and `fftfreq` put the frequency axis in order.

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

Chemical shifts follow the electronic environment: hydrogens on carbons next to electronegative atoms or π systems are *deshielded* and appear at higher δ. A crude but instructive predictor assigns each hydrogen-bearing carbon a base shift by its environment, using SMARTS patterns (Chapter 21) checked in order; the first pattern that matches an atom wins, which is what `setdefault` does.

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

A mass spectrometer measures the mass-to-charge ratio m/z of ions. Most elements have several stable **isotopes**: about 1.1 % of carbon is ¹³C, and chlorine is roughly three parts ³⁵Cl to one part ³⁷Cl. A molecule therefore appears not as one peak but as a cluster: the M peak with only the lightest isotopes, an M+1 peak with one heavier isotope, and so on. The shape of the cluster is a fingerprint of the elements present. The probability of each combination follows from multiplying the probabilities of each atom's isotope, so the pattern of a molecule is built by combining its atoms one at a time, a process called **convolution**. For a single chlorine atom the pattern is just its natural abundances:

```{code-cell} ipython3
cl = {35: 0.7576, 37: 0.2424}
print({m: round(100 * p / max(cl.values()), 1) for m, p in cl.items()})    # M : M+2 ≈ 100 : 32
```

The general function adds atoms one by one: for every peak in the current distribution and every isotope of the next atom, it shifts the mass and multiplies the probabilities. Peaks are tracked by their nominal (integer) mass offset, and tiny probabilities are dropped to keep the calculation fast.

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

```{admonition} Summary
:class: tip
- Beer–Lambert: $A = \varepsilon l c$; a calibration line of standards converts absorbance to concentration.
- Transmittance $T = 10^{-A}$ predicts perceived colour.
- Asymmetric least squares removes curved baselines; `find_peaks` with a prominence threshold picks bands.
- First-order NMR multiplets follow the n + 1 rule with Pascal's-triangle intensities; $J$ is fixed in Hz.
- The FFT turns an NMR free induction decay into a spectrum.
- Isotope patterns are convolutions of each atom's isotope abundances.
```

## Exercises

**★ 1.** Use the calibration line to find the concentrations of solutions with A = 0.25, 0.48 and 0.95.

**★★ 2.** Simulate the ¹H NMR spectrum of ethyl acetate at 60 MHz and at 600 MHz. What changes, and why?

**★★★ 3.** Fit three Lorentzians to the overlapping C–H stretches of the corrected IR spectrum and report their areas.
