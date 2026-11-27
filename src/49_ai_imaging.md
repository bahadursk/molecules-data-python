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

Electron micrographs, optical images and hyperspectral maps are all arrays of numbers, and the questions chemists ask of them (how big are the particles? which phase is this region? where is the defect?) are image-analysis problems. Deep learning now does much of this work; segmentation architectures such as U-Net {cite:p}`ronneberger2015` are standard, and the use of deep learning in electron microscopy is thoroughly reviewed by Ede {cite:p}`ede2021`. The reason to automate is throughput and consistency. A particle size distribution measured by hand from a few dozen particles on one micrograph carries a large uncertainty and depends on who drew the outlines; a pipeline that processes a hundred images the same way every time gives a number that can be reproduced. Two tasks recur throughout. **Segmentation** decides, for every pixel, which object or phase it belongs to, which is what turns an image into measurements of size, shape and coverage. **Unmixing** applies to data where every pixel holds a whole spectrum and the question is which chemical components are present and where. This chapter does both, first with classical image processing, then with a small neural network, and ends with denoising and the warning that goes with it.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
rng = np.random.default_rng(7)
```

## An image is an array

A greyscale image is a two-dimensional NumPy array (Chapter 12) whose values are brightnesses, so every operation in this chapter is array arithmetic. The synthetic micrograph below is built that way: bright discs on a grey background, blurred to imitate the microscope's limited resolution, with an illumination gradient across the field and random noise added. Because it is synthetic, the true label of every pixel is known, which is what makes it possible to score the methods honestly. A disc is drawn by testing which pixel coordinates lie within a radius of its centre, using the coordinate grids from `np.mgrid`:

```{code-cell} ipython3
yy, xx = np.mgrid[:5, :5]
print(((yy - 2) ** 2 + (xx - 2) ** 2 < 2 ** 2).astype(int))
```

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

The classical pipeline has four steps, and each one has a reason. **Flattening** subtracts a heavily smoothed copy of the image, which removes the slow illumination gradient while leaving the particles, since they are much smaller than the smoothing window. **Thresholding** turns the image into a yes/no mask; **Otsu's method** chooses the threshold automatically by picking the grey level that splits the histogram into two groups as distinct as possible, that is, the level that maximises the variance between the groups. A **binary opening** then erodes and dilates the mask, which deletes isolated noisy pixels while leaving real objects. Finally `ndi.label` numbers the connected regions, so that each particle becomes a separate integer:

```{code-cell} ipython3
mask_demo = np.array([[1, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 1]])
print(ndi.label(mask_demo)[0])          # two separate blobs get labels 1 and 2
```

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

Labelling connects anything that touches, so two particles in contact become one object and the measured size is wrong. The **watershed** method separates them using the **distance transform**, which replaces each pixel of the mask by its distance to the nearest background pixel. Each particle then has a peak at its centre, and the valley between two touching particles marks the boundary. Local maxima of the distance become **markers**, one per particle, and the mask is filled outward from the markers in order of decreasing distance, as if water were rising from each centre, with boundaries where two floods meet. The distance transform of a small block rises towards its centre line:

```{code-cell} ipython3
block = np.zeros((5, 9)); block[1:4, 1:8] = 1
print(ndi.distance_transform_edt(block).round(1))
```

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

Once each particle has its own label, its area is just the number of pixels carrying that label, and an equivalent radius follows from $A = \pi r^2$. A quality check comes for free. **Circularity**, $4\pi A/P^2$ where $P$ is the perimeter, equals 1 for a perfect circle and falls as a shape becomes elongated or ragged, so it identifies objects that the watershed failed to split or that run off the edge of the image. The perimeter is counted as the pixels of the object that are lost when it is eroded by one pixel:

```{code-cell} ipython3
A_demo, P_demo = 1000, 112          # a roughly circular object
print(round(4 * np.pi * A_demo / P_demo ** 2, 2))
```

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

A learned segmenter looks at a patch around each pixel and decides whether that pixel belongs to a particle. Here is a compact convolutional network, trained on synthetic images with known labels, which is exactly how microscopy segmentation models are usually bootstrapped. A **convolutional** layer slides a small learnable filter across the image and reports how strongly each position matches it, which is how a network learns to detect edges, blobs and textures (Chapter 34). The **U-Net** shape has two halves: an encoder that downsamples the image, so that later filters see a larger area and can use context, and a decoder that upsamples back to full resolution so that an output is produced for every pixel. The **skip connection** passes the high-resolution features from the encoder directly to the decoder, which is what lets the output have sharp boundaries rather than blurry ones. The loss is **binary cross-entropy** on each pixel, comparing the predicted probability of "particle" with the known label.

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

Segmentations are compared by the **intersection over union** (IoU), also called the Jaccard index: the number of pixels both the prediction and the truth call "particle", divided by the number either of them does. It is 1 for a perfect match and 0 for no overlap, and unlike plain accuracy it is not flattered by a mostly empty image. On two overlapping blocks:

```{code-cell} ipython3
a_demo = np.array([1, 1, 1, 0, 0], bool)
b_demo = np.array([0, 1, 1, 1, 0], bool)
print((a_demo & b_demo).sum() / (a_demo | b_demo).sum())
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

The network reaches an IoU of about 0.98 against 0.92 for the classical pipeline. The margin is modest, because the classical pipeline was given its flattening step and a threshold chosen automatically; the real advantage is that the network found the equivalent of those steps by itself and needs no hand-tuning for a new image. In return it needs labelled data, which in real microscopy usually comes from a mixture of hand annotation, simulation and self-training.

## Hyperspectral imaging: a spectrum at every pixel

A Raman or energy-dispersive X-ray map is a three-dimensional data cube: two spatial axes and one spectral. Unmixing it recovers both the phases present and where they are. Each pixel's spectrum is a mixture of the spectra of the phases that lie under it, weighted by how much of each is there, exactly the linear mixing of Chapter 39 but repeated across an image. **Unmixing** means recovering both the pure spectra and the weight maps from the cube alone, with no reference spectra supplied. Building the cube is the forward version of that: `np.einsum` multiplies the abundance of each phase at each pixel by that phase's spectrum and sums over phases, giving an array of shape (y, x, bands):

```{code-cell} ipython3
ab_demo = np.array([[[0.5]], [[0.5]]])                      # two phases, one pixel, half each
sp_demo = np.array([[1.0, 0.0], [0.0, 2.0]])                # their spectra, two bands
print(np.einsum("pyx,pb->yxb", ab_demo, sp_demo))
```

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

**Non-negative matrix factorisation** (NMF) splits the flattened cube, one row per pixel, into two matrices whose product approximates it: `W`, how much of each component is in each pixel, and `H`, the spectrum of each component. The constraint that makes it work for chemistry is in the name: both factors must be non-negative, since neither a concentration nor an intensity can be negative, and that is enough to make the recovered components resemble real pure spectra. PCA (Chapter 33) would fit the data just as well but with components containing negative lobes that mean nothing chemically. Two things must be checked. The number of components is chosen by the user, and a wrong choice either merges phases or splits one phase in two; and the components come out in no particular order, so they have to be matched against the truth, here by correlating each recovered spectrum with each true one.

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

If reference spectra or labelled regions are available, the map can be made by classification instead: treat each pixel's spectrum as a feature vector and each phase as a class. This is supervised, so it needs labels, but it handles overlapping spectra better than unmixing and gives a probability per pixel. The errors concentrate at the boundaries between phases, where a pixel genuinely contains two phases at once and no single label is right.

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

Electron and X-ray doses damage samples, so real micrographs are noisy by necessity. Denoising lets you work at lower dose. Three filters are compared, in increasing order of cleverness. A **Gaussian filter** averages each pixel with its neighbours, which suppresses noise and blurs edges in equal measure. A **median filter** takes the middle value of the neighbourhood instead, which removes isolated bright or dark pixels while keeping edges sharper. **Non-local means** averages a pixel with other pixels anywhere nearby whose *surroundings* look similar, so it averages over genuinely equivalent parts of the image rather than over an arbitrary disc. Quality is measured by the **peak signal-to-noise ratio** (PSNR) in decibels against the clean image, where higher is better and a few decibels is a visible difference. Because it is a logarithmic measure, halving the noise adds about 6 dB:

```{code-cell} ipython3
for noise in [0.4, 0.2, 0.1]:
    print(noise, round(10 * np.log10(1.0 / noise ** 2), 1), "dB")
```

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

```{admonition} Summary
:class: tip
- Images are arrays; flatten the illumination, threshold (Otsu), clean up, then label connected regions.
- Watershed on the distance transform splits touching particles; circularity flags the ones it missed.
- Document every pipeline choice, because they shift a reported particle size by tens of per cent.
- A small U-Net segments pixels directly and copes with gradients and noise; compare with IoU.
- NMF unmixes a hyperspectral cube into pure spectra and abundance maps without reference spectra.
- Denoising filters trade sharpness for noise; never trust a feature that is not in the raw data.
```

## Reading further

- U-Net {cite:p}`ronneberger2015` is the architecture most microscopy segmentation still builds on.
- Ede's review of deep learning in electron microscopy {cite:p}`ede2021` covers denoising, super-resolution, segmentation and the pitfalls above.

## Exercises

**★ 1.** Raise the noise level in `synth_micrograph` until the classical threshold fails. Does the CNN still work?

**★★ 2.** Add a second particle population with a different mean radius and check whether the size distribution shows two peaks.

**★★★ 3.** Train the segmenter on images with one noise level and test it on another. Then add noise augmentation during training and measure the improvement.
