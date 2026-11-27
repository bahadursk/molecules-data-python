---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# AI for Chemical and Microscopy Imaging

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 12, 13, 31 and 34.
```

Electron micrographs, optical images and hyperspectral maps are all arrays of numbers, and the questions chemists ask of them (how big are the particles? which phase is this region? where is the defect?) are image-analysis problems. Deep learning now does much of this work; segmentation architectures such as U-Net {cite:p}`ronneberger2015` are standard, and the use of deep learning in electron microscopy is thoroughly reviewed by Ede {cite:p}`ede2021`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
rng = np.random.default_rng(7)
```

## An image is an array

```{code-cell} ipython3
def synth_micrograph(n=320, n_particles=45, r_mean=13, r_sd=4.5, noise=0.10, blur=1.2, seed=0):
    """A synthetic TEM-like image: bright circular nanoparticles on a noisy grey background."""
    r = np.random.default_rng(seed)
    img = np.zeros((n, n))
    truth = np.zeros((n, n), dtype=int)
    radii = []
    yy, xx = np.mgrid[:n, :n]
    for k in range(n_particles):
        rad = max(4.0, r.normal(r_mean, r_sd))
        cy, cx = r.uniform(rad, n - rad, 2)
        mask = (yy - cy) ** 2 + (xx - cx) ** 2 < rad ** 2
        img[mask] = 1.0
        truth[mask] = k + 1
        radii.append(rad)
    img = ndi.gaussian_filter(img, blur)
    img += 0.25 + 0.12 * (xx / n)                                 # uneven illumination
    img += r.normal(0, noise, img.shape)
    return img, truth, np.array(radii)

img, truth, radii = synth_micrograph()
fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].imshow(img, cmap="gray"); axes[0].set_title(f"synthetic micrograph {img.shape}")
axes[1].imshow(truth, cmap="nipy_spectral"); axes[1].set_title("ground truth labels")
axes[2].hist(img.ravel(), bins=80, color="#6c4ad6"); axes[2].set(title="intensity histogram", xlabel="grey level")
for a in axes[:2]:
    a.axis("off")
plt.tight_layout(); plt.show()
```

## Classical segmentation: threshold, label, measure

```{code-cell} ipython3
def otsu(image, bins=256):
    """Otsu's threshold: maximise between-class variance."""
    hist, edges = np.histogram(image, bins=bins)
    p = hist / hist.sum()
    centres = 0.5 * (edges[1:] + edges[:-1])
    omega = np.cumsum(p)
    mu = np.cumsum(p * centres)
    mu_t = mu[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / (omega * (1 - omega))
    return centres[np.nanargmax(sigma_b)]

flat = img - ndi.uniform_filter(img, 60)                    # flatten the illumination gradient
t = otsu(flat)
binary = ndi.binary_opening(flat > t, np.ones((3, 3)))
labels, n_found = ndi.label(binary)
print(f"Otsu threshold {t:.3f}; found {n_found} objects (truth {truth.max()})")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (im, title, cmap) in zip(axes, [(img, "raw", "gray"), (flat, "flattened", "gray"),
                                        (binary, f"binary (t = {t:.2f})", "gray"), (labels, "labelled", "nipy_spectral")]):
    ax.imshow(im, cmap=cmap); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

### Watershed: splitting touching particles

```{code-cell} ipython3
distance = ndi.distance_transform_edt(binary)
local_max = (distance == ndi.maximum_filter(distance, size=9)) & (distance > 4)
markers, n_markers = ndi.label(local_max)

def watershed_simple(distance, markers, mask):
    """Flood from the markers in order of decreasing distance (a compact watershed)."""
    out = markers.copy()
    order = np.argsort(-distance[mask])
    coords = np.array(np.nonzero(mask)).T[order]
    changed = True
    while changed:
        changed = False
        for y, x in coords:
            if out[y, x] == 0:
                nb = out[max(0, y - 1):y + 2, max(0, x - 1):x + 2]
                vals = nb[nb > 0]
                if vals.size:
                    out[y, x] = np.bincount(vals).argmax()
                    changed = True
    return out

ws = watershed_simple(distance, markers, binary)
print(f"watershed separated {ws.max()} particles")

fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
axes[0].imshow(distance, cmap="magma"); axes[0].set_title("distance transform")
axes[1].imshow(ws, cmap="nipy_spectral"); axes[1].set_title(f"watershed: {ws.max()} particles")
axes[2].imshow(img, cmap="gray")
axes[2].contour(ws > 0, levels=[0.5], colors="#0f9d8a", linewidths=0.8)
axes[2].set_title("outlines on the original")
for a in axes:
    a.axis("off")
plt.tight_layout(); plt.show()
```

## Particle size distribution

```{code-cell} ipython3
areas = np.array([(ws == k).sum() for k in range(1, ws.max() + 1)])
areas = areas[areas > 30]
r_measured = np.sqrt(areas / np.pi)
fig, axes = plt.subplots(1, 2, figsize=(11, 4))
axes[0].hist(radii, bins=16, alpha=0.6, color="#6c4ad6", label=f"truth (mean {radii.mean():.1f} px)")
axes[0].hist(r_measured, bins=16, alpha=0.6, color="#f2a93b", label=f"measured (mean {r_measured.mean():.1f} px)")
axes[0].set(xlabel="radius / pixels", ylabel="count"); axes[0].legend()
axes[0].set_title("Size distribution recovered from the image")
# A circularity check catches merged or clipped objects
circ = []
for k in range(1, ws.max() + 1):
    m = ws == k
    if m.sum() <= 30:
        continue
    perim = np.sum(m & ~ndi.binary_erosion(m))
    circ.append(4 * np.pi * m.sum() / max(perim ** 2, 1))
axes[1].scatter(r_measured[:len(circ)], circ, s=30, color="#0f9d8a")
axes[1].axhline(0.7, ls="--", color="#e5487a"); axes[1].set(xlabel="radius / px", ylabel="circularity",
                                                            title="Low circularity = merged particles, reject them")
plt.tight_layout(); plt.show()
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Always report how a size distribution was obtained: the threshold method, the minimum object size, whether touching particles were split, and whether clipped particles at the image edge were excluded. Different choices shift a mean particle size by tens of per cent, and nobody can reproduce a number whose pipeline is undocumented.
```

## A small convolutional network for segmentation

A learned segmenter looks at a patch around each pixel and decides whether that pixel belongs to a particle. Here is a compact convolutional network, trained on synthetic images with known labels, which is exactly how microscopy segmentation models are usually bootstrapped.

```{code-cell} ipython3
import torch
import torch.nn as nn
torch.manual_seed(0)

train_imgs, train_masks = [], []
for s in range(8):
    im, tr, _ = synth_micrograph(n=160, n_particles=18, noise=0.14, seed=100 + s)
    train_imgs.append(im); train_masks.append((tr > 0).astype(np.float32))
Xtr = torch.tensor(np.stack(train_imgs)[:, None], dtype=torch.float32)
Ytr = torch.tensor(np.stack(train_masks)[:, None], dtype=torch.float32)

class TinyUNet(nn.Module):
    """A miniature encoder–decoder with one skip connection, in the spirit of U-Net."""
    def __init__(self, c=16):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(1, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, c, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.MaxPool2d(2), nn.Conv2d(c, 2 * c, 3, padding=1), nn.ReLU(),
                                  nn.Conv2d(2 * c, 2 * c, 3, padding=1), nn.ReLU())
        self.up = nn.ConvTranspose2d(2 * c, c, 2, stride=2)
        self.dec = nn.Sequential(nn.Conv2d(2 * c, c, 3, padding=1), nn.ReLU(), nn.Conv2d(c, 1, 1))
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        d = self.up(e2)
        return self.dec(torch.cat([d, e1], dim=1))

net = TinyUNet()
opt = torch.optim.AdamW(net.parameters(), lr=3e-3)
lossf = nn.BCEWithLogitsLoss()
curve = []
for epoch in range(120):
    net.train()
    loss = lossf(net(Xtr), Ytr)
    opt.zero_grad(); loss.backward(); opt.step()
    curve.append(loss.item())
plt.plot(curve, color="#6c4ad6"); plt.xlabel("epoch"); plt.ylabel("BCE loss"); plt.title("Training the segmenter"); plt.show()
```

```{code-cell} ipython3
test_img, test_truth, _ = synth_micrograph(n=160, n_particles=18, noise=0.16, seed=999)
net.eval()
with torch.no_grad():
    prob = torch.sigmoid(net(torch.tensor(test_img[None, None], dtype=torch.float32)))[0, 0].numpy()
pred_mask = prob > 0.5
truth_mask = test_truth > 0

classical = ndi.binary_opening(test_img - ndi.uniform_filter(test_img, 60) > otsu(test_img - ndi.uniform_filter(test_img, 60)),
                               np.ones((3, 3)))
def iou(a, b):
    return (a & b).sum() / (a | b).sum()
print(f"IoU  classical threshold {iou(classical, truth_mask):.3f}   CNN {iou(pred_mask, truth_mask):.3f}")

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for ax, (im, title, cmap) in zip(axes, [(test_img, "test image", "gray"), (truth_mask, "truth", "gray"),
                                        (prob, "CNN probability", "magma"), (pred_mask ^ truth_mask, "CNN errors", "Reds")]):
    ax.imshow(im, cmap=cmap); ax.set_title(title); ax.axis("off")
plt.tight_layout(); plt.show()
```

The network learns to ignore the illumination gradient and the noise, which the global threshold cannot, and it needs no hand-tuning per image. In return it needs labelled data, which in real microscopy usually comes from a mixture of hand annotation, simulation and self-training.

## Hyperspectral imaging: a spectrum at every pixel

A Raman or energy-dispersive X-ray map is a three-dimensional data cube: two spatial axes and one spectral. Unmixing it recovers both the phases present and where they are.

```{code-cell} ipython3
n, bands = 96, 120
wl = np.linspace(200, 1800, bands)
def peak(c, w, h):
    return h * np.exp(-0.5 * ((wl - c) / w) ** 2)
phases = {"phase A": peak(520, 18, 1.0) + peak(980, 25, 0.4),
          "phase B": peak(1080, 20, 0.9) + peak(700, 30, 0.3),
          "phase C": peak(1580, 24, 1.0) + peak(1350, 40, 0.6)}

yy, xx = np.mgrid[:n, :n]
abundance = np.stack([
    np.clip(1.2 - np.hypot(yy - 30, xx - 30) / 28, 0, 1),
    np.clip(1.2 - np.hypot(yy - 65, xx - 40) / 24, 0, 1),
    np.clip((xx > 55).astype(float) * (0.6 + 0.4 * yy / n), 0, 1)])
abundance /= abundance.sum(0, keepdims=True) + 1e-9
cube = np.einsum("pyx,pb->yxb", abundance, np.stack(list(phases.values()))) + rng.normal(0, 0.03, (n, n, bands))

fig, axes = plt.subplots(1, 4, figsize=(16, 3.8))
for ax, (name, sp), c in zip(axes[:3], phases.items(), ["#6c4ad6", "#0f9d8a", "#e5487a"]):
    ax.plot(wl, sp, color=c); ax.set(title=name, xlabel="Raman shift / cm$^{-1}$")
axes[3].imshow(np.moveaxis(abundance, 0, -1)); axes[3].set_title("true phase map (RGB)"); axes[3].axis("off")
plt.tight_layout(); plt.show()
```

```{code-cell} ipython3
from sklearn.decomposition import NMF, PCA
flat_cube = cube.reshape(-1, bands)
nmf = NMF(n_components=3, init="nndsvda", max_iter=600, random_state=0)
W = nmf.fit_transform(np.clip(flat_cube, 0, None))
H = nmf.components_

fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
for k in range(3):
    axes[0, k].plot(wl, H[k], color="k"); axes[0, k].set_title(f"recovered component {k + 1}")
    im = axes[1, k].imshow(W[:, k].reshape(n, n), cmap="viridis"); axes[1, k].axis("off")
    axes[1, k].set_title(f"abundance map {k + 1}")
plt.tight_layout(); plt.show()

# Match recovered components to the truth by spectral correlation
corr = np.array([[np.corrcoef(H[k], sp)[0, 1] for sp in phases.values()] for k in range(3)])
print(pd.DataFrame(corr, index=[f"component {k+1}" for k in range(3)], columns=list(phases)).round(2))
```

Non-negative matrix factorisation recovers both the pure spectra and their spatial distribution without being told what the phases are: **blind source separation**, the workhorse of hyperspectral chemistry.

### Pixel-wise classification

```{code-cell} ipython3
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

y_pix = abundance.argmax(0).ravel()
Xa, Xb, ya, yb = train_test_split(flat_cube, y_pix, test_size=0.7, random_state=0, stratify=y_pix)
clf = RandomForestClassifier(200, n_jobs=-1, random_state=0).fit(Xa, ya)
pred_map = clf.predict(flat_cube).reshape(n, n)
print(f"pixel classification accuracy {accuracy_score(yb, clf.predict(Xb)):.2%}")
fig, axes = plt.subplots(1, 3, figsize=(13, 4))
axes[0].imshow(abundance.argmax(0), cmap="Set2"); axes[0].set_title("dominant phase (truth)")
axes[1].imshow(pred_map, cmap="Set2"); axes[1].set_title("predicted")
axes[2].imshow(pred_map != abundance.argmax(0), cmap="Reds"); axes[2].set_title("errors are at the boundaries")
for a in axes:
    a.axis("off")
plt.tight_layout(); plt.show()
```

## Denoising: more signal from less dose

Electron and X-ray doses damage samples, so real micrographs are noisy by necessity. Denoising lets you work at lower dose, and the simplest learned denoiser is remarkably effective.

```{code-cell} ipython3
clean, _, _ = synth_micrograph(n=160, n_particles=20, noise=0.0, seed=5)
noisy = clean + rng.normal(0, 0.35, clean.shape)
gauss = ndi.gaussian_filter(noisy, 2)
median = ndi.median_filter(noisy, 5)

def nlmeans_lite(im, h=0.4, patch=3, window=7):
    """A small non-local means: average pixels with similar neighbourhoods."""
    pad = window // 2 + patch // 2
    P = np.pad(im, pad, mode="reflect")
    out = np.zeros_like(im)
    kernel = np.ones((patch, patch)) / patch ** 2
    sq = ndi.uniform_filter(P ** 2, patch)
    mean = ndi.uniform_filter(P, patch)
    for dy in range(-(window // 2), window // 2 + 1):
        for dx in range(-(window // 2), window // 2 + 1):
            shifted = np.roll(np.roll(mean, dy, 0), dx, 1)
            d2 = (mean - shifted) ** 2
            w = np.exp(-d2 / h ** 2)
            out += (w * np.roll(np.roll(P, dy, 0), dx, 1))[pad:-pad, pad:-pad]
    norm = np.zeros_like(im)
    for dy in range(-(window // 2), window // 2 + 1):
        for dx in range(-(window // 2), window // 2 + 1):
            shifted = np.roll(np.roll(mean, dy, 0), dx, 1)
            norm += np.exp(-((mean - shifted) ** 2) / h ** 2)[pad:-pad, pad:-pad]
    return out / norm

nlm = nlmeans_lite(noisy)
def psnr(a, b):
    return 10 * np.log10(1.0 / np.mean((a - b) ** 2))
fig, axes = plt.subplots(1, 5, figsize=(17, 3.6))
for ax, (im, title) in zip(axes, [(clean, "clean"), (noisy, f"noisy ({psnr(noisy, clean):.1f} dB)"),
                                  (gauss, f"gaussian ({psnr(gauss, clean):.1f} dB)"),
                                  (median, f"median ({psnr(median, clean):.1f} dB)"),
                                  (nlm, f"non-local means ({psnr(nlm, clean):.1f} dB)")]):
    ax.imshow(im, cmap="gray", vmin=0, vmax=1.4); ax.set_title(title, fontsize=10); ax.axis("off")
plt.tight_layout(); plt.show()
```

```{warning}
Denoising and super-resolution models can **invent** structure that was never in the sample: a lattice where there was noise, a particle where there was none. Any feature that matters must be visible in the raw data, or confirmed at higher dose. Publish the raw frames alongside the processed ones.
```

## Reading further

- U-Net {cite:p}`ronneberger2015` is the architecture most microscopy segmentation still builds on.
- Ede's review of deep learning in electron microscopy {cite:p}`ede2021` covers denoising, super-resolution, segmentation and the pitfalls above.

## Exercises

**★ 1.** Raise the noise level in `synth_micrograph` until the classical threshold fails. Does the CNN still work?

**★★ 2.** Add a second particle population with a different mean radius and check whether the size distribution shows two peaks.

**★★★ 3.** Train the segmenter on images with one noise level and test it on another. Then add noise augmentation during training and measure the improvement.
