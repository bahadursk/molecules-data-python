"""Generate the book's logo and cover art (original artwork, CC-BY-4.0 like the rest of the book).

    python tools/make_art.py

Writes book/_static/logo.png, book/_static/cover.png and book/_static/social_card.png.
Everything is drawn from scratch with Matplotlib and RDKit; no external images are used.
"""
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon, Circle, FancyBboxPatch
from rdkit import Chem
from rdkit.Chem import AllChem

OUT = Path(__file__).resolve().parents[1] / "book" / "_static"
OUT.mkdir(parents=True, exist_ok=True)

TEAL, VIOLET, AMBER, ROSE = "#0f9d8a", "#6c4ad6", "#f2a93b", "#e5487a"
INK, PAPER = "#14142b", "#fbfaf7"
PALETTE = [TEAL, VIOLET, AMBER, ROSE]


def mol_2d(smiles):
    m = Chem.MolFromSmiles(smiles)
    AllChem.Compute2DCoords(m)
    xy = m.GetConformer().GetPositions()[:, :2]
    xy -= xy.mean(0)
    return m, xy


def draw_mol(ax, smiles, centre, scale, colour, lw=2.0, alpha=1.0, atom_dots=True):
    m, xy = mol_2d(smiles)
    xy = xy * scale + np.asarray(centre)
    for b in m.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        (x1, y1), (x2, y2) = xy[i], xy[j]
        order = b.GetBondTypeAsDouble()
        if order >= 2 and not b.GetIsAromatic():
            dx, dy = y2 - y1, x1 - x2
            n = np.hypot(dx, dy) or 1
            off = 0.09 * scale
            for s in (-1, 1):
                ax.plot([x1 + s * off * dx / n, x2 + s * off * dx / n],
                        [y1 + s * off * dy / n, y2 + s * off * dy / n],
                        color=colour, lw=lw * 0.8, alpha=alpha, solid_capstyle="round")
        else:
            ax.plot([x1, x2], [y1, y2], color=colour, lw=lw, alpha=alpha, solid_capstyle="round")
    if atom_dots:
        for a, (x, y) in zip(m.GetAtoms(), xy):
            if a.GetSymbol() != "C":
                c = {"O": ROSE, "N": VIOLET, "S": AMBER, "F": TEAL, "Cl": TEAL}.get(a.GetSymbol(), INK)
                ax.add_patch(Circle((x, y), 0.16 * scale, color=c, alpha=alpha, zorder=5))
    # aromatic rings: draw an inner circle
    ri = m.GetRingInfo()
    for ring in ri.AtomRings():
        if all(m.GetAtomWithIdx(k).GetIsAromatic() for k in ring):
            c = xy[list(ring)].mean(0)
            ax.add_patch(Circle(c, 0.55 * scale, fill=False, ec=colour, lw=lw * 0.7, alpha=alpha * 0.8))


def logo():
    fig, ax = plt.subplots(figsize=(4, 4), dpi=150)
    fig.patch.set_alpha(0)
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.3); ax.set_aspect("equal"); ax.axis("off")
    hexagon = RegularPolygon((0, 0), 6, radius=1.0, orientation=np.pi / 6,
                             fill=False, ec=TEAL, lw=10, joinstyle="round")
    ax.add_patch(hexagon)
    ax.add_patch(Circle((0, 0), 0.58, fill=False, ec=VIOLET, lw=6))
    # vertices as data points, coloured like a scatter plot
    ang = np.pi / 6 + np.arange(6) * np.pi / 3
    for k, a in enumerate(ang):
        ax.add_patch(Circle((np.cos(a), np.sin(a)), 0.16, color=PALETTE[k % 4], zorder=5))
    # a rising "learning curve" through the ring
    x = np.linspace(-0.45, 0.45, 50)
    ax.plot(x, 0.35 * np.tanh(4 * x) - 0.02, color=AMBER, lw=5, solid_capstyle="round")
    fig.savefig(OUT / "logo.png", transparent=True, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


COVER_MOLS = [
    ("CC(=O)Oc1ccccc1C(=O)O", TEAL),                          # aspirin
    ("Cn1cnc2c1c(=O)n(C)c(=O)n2C", VIOLET),                   # caffeine
    ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", ROSE),                     # ibuprofen
    ("O=C(O)c1ccncc1", AMBER),                                # isonicotinic acid
    ("c1ccc2cc3ccccc3cc2c1", TEAL),                           # anthracene
    ("OC1C(O)C(O)C(CO)OC1O", VIOLET),                         # a pyranose
    ("CC(=O)Nc1ccc(O)cc1", ROSE),                             # paracetamol
    ("C1=CC=C2C(=C1)C=CC=C2", AMBER),                         # naphthalene (Kekulé)
]


def cover(w=8.0, h=12.0, name="cover.png", dpi=200):
    rng = np.random.default_rng(2026)
    fig = plt.figure(figsize=(w, h), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w); ax.set_ylim(0, h); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), w, h, color=INK))

    # chemical-space scatter in the background
    n = 900
    cl = rng.integers(0, 4, n)
    centres = np.array([[2.0, 4.2], [5.6, 3.4], [3.6, 6.6], [6.2, 6.2]])
    pts = centres[cl] + rng.normal(0, 0.75, (n, 2))
    sizes = rng.uniform(4, 40, n)
    for k in range(4):
        m = cl == k
        ax.scatter(pts[m, 0], pts[m, 1], s=sizes[m], color=PALETTE[k], alpha=0.28, lw=0)

    # molecules floating over the scatter
    spots = [(1.6, 8.4), (6.1, 8.9), (3.4, 4.6), (1.5, 2.3), (5.9, 1.9), (6.6, 6.0), (1.7, 6.3), (3.4, 3.0)]
    for (smi, col), c in zip(COVER_MOLS, spots):
        draw_mol(ax, smi, c, scale=0.36, colour=col, lw=2.6, alpha=0.95)

    # a fitted curve (data meets models)
    xs = np.linspace(0.6, w - 0.6, 300)
    ax.plot(xs, 7.4 + 0.55 * np.sin(xs * 1.25) * np.exp(-0.05 * xs), color=PAPER, lw=1.5, alpha=0.5)

    # title block
    ax.add_patch(FancyBboxPatch((0.45, 9.55), w - 0.9, 2.05, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=PAPER, ec="none", alpha=0.96))
    ax.text(w / 2, 11.05, "Molecules, Data", ha="center", va="center", fontsize=34, weight="bold", color=INK)
    ax.text(w / 2, 10.3, "and Python", ha="center", va="center", fontsize=34, weight="bold", color=TEAL)
    ax.text(w / 2, 9.8, "From school chemistry to AI-driven research", ha="center", va="center",
            fontsize=11, color=VIOLET, style="italic")

    # level ribbon
    for k, (lbl, c) in enumerate([("Python", TEAL), ("Cheminformatics", VIOLET),
                                  ("Machine learning", AMBER), ("Research frontiers", ROSE)]):
        x0 = 0.45 + k * (w - 0.9) / 4
        ax.add_patch(plt.Rectangle((x0, 0.95), (w - 0.9) / 4 - 0.06, 0.32, color=c))
        ax.text(x0 + ((w - 0.9) / 4 - 0.06) / 2, 1.11, lbl, ha="center", va="center", fontsize=8.5,
                color="white", weight="bold")
    ax.text(w / 2, 0.5, "Dr. Bahadur Sk", ha="center", va="center", fontsize=17, color=PAPER, weight="bold")
    fig.savefig(OUT / name, dpi=dpi, facecolor=INK)
    plt.close(fig)


def social_card():
    """1200 × 630 card used when the book's link is shared."""
    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 12); ax.set_ylim(0, 6.3); ax.axis("off")
    ax.add_patch(plt.Rectangle((0, 0), 12, 6.3, color=INK))
    rng = np.random.default_rng(7)
    p = rng.normal([8.8, 3.1], [1.4, 1.2], (500, 2))
    ax.scatter(p[:, 0], p[:, 1], s=rng.uniform(5, 40, 500), c=rng.integers(0, 4, 500),
               cmap=matplotlib.colors.ListedColormap(PALETTE), alpha=0.3, lw=0)
    for (smi, col), c in zip(COVER_MOLS[:4], [(7.6, 4.7), (10.4, 4.2), (8.0, 1.9), (10.7, 1.4)]):
        draw_mol(ax, smi, c, scale=0.33, colour=col, lw=2.4)
    ax.text(0.7, 4.3, "Molecules, Data", fontsize=44, weight="bold", color=PAPER, va="center")
    ax.text(0.7, 3.3, "and Python", fontsize=44, weight="bold", color=TEAL, va="center")
    ax.text(0.7, 2.3, "An interactive book: Python, cheminformatics\nand machine learning for chemists",
            fontsize=15, color=PAPER, va="center", alpha=0.9)
    ax.text(0.7, 0.9, "Dr. Bahadur Sk", fontsize=17, color=AMBER, weight="bold", va="center")
    fig.savefig(OUT / "social_card.png", dpi=100, facecolor=INK)
    plt.close(fig)


if __name__ == "__main__":
    logo()
    cover()
    social_card()
    for f in ["logo.png", "cover.png", "social_card.png"]:
        print("wrote", (OUT / f).relative_to(OUT.parents[1]))
