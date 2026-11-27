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


# ---------------------------------------------------------------------------
# "Periodic table" of the Python libraries used in the book (original artwork:
# text tiles only, no third-party logos).
LIB_TILES = [
    # (row, col, symbol, name, category, chapter)
    (0, 0, "Py", "Python", "core", 1),
    (0, 17, "Ju", "Jupyter", "core", 1),
    (1, 0, "Np", "NumPy", "scipy", 12), (1, 1, "Pd", "pandas", "scipy", 15),
    (1, 12, "Rd", "RDKit", "chem", 19), (1, 13, "3D", "py3Dmol", "chem", 23),
    (1, 14, "Pc", "PubChemPy", "chem", 24), (1, 15, "Ch", "ChEMBL", "chem", 24),
    (1, 16, "Pdb", "PDB", "chem", 24), (1, 17, "Mt", "Materials\nProject", "chem", 24),
    (2, 0, "Mp", "Matplotlib", "viz", 13), (2, 1, "Sb", "seaborn", "viz", 14),
    (2, 12, "Sc", "SciPy", "scipy", 16), (2, 13, "Sy", "SymPy", "scipy", 17),
    (2, 14, "Pi", "pint", "scipy", 18), (2, 15, "Un", "uncert-\nainties", "scipy", 18),
    (2, 16, "Ps", "PySCF", "chem", 28), (2, 17, "As", "ASE", "chem", 30),
    (3, 0, "Pl", "Plotly", "viz", 14), (3, 1, "Pg", "pymatgen", "chem", 30),
    (3, 2, "Md", "MDAnalysis", "chem", 29), (3, 3, "Sk", "scikit-\nlearn", "ml", 31),
    (3, 4, "Pt", "PyTorch", "ml", 34), (3, 5, "Gp", "Gaussian\nprocesses", "ml", 35),
    (3, 6, "Bo", "Bayesian\nopt.", "ml", 35), (3, 7, "Gn", "Graph\nnets", "ml", 34),
    (3, 8, "Tf", "Trans-\nformers", "ml", 34), (3, 9, "Ga", "Genetic\nalgorithms", "ml", 45),
    (3, 10, "Pls", "PLS", "ml", 39), (3, 11, "Nmf", "NMF", "ml", 49),
    (3, 12, "Un", "U-Net", "ml", 49), (3, 13, "Sh", "Shapley\nvalues", "ml", 51),
    (3, 14, "Al", "Active\nlearning", "ml", 35), (3, 15, "Dt", "Digital\ntwins", "ml", 50),
    (3, 16, "Ll", "LLMs", "ml", 46), (3, 17, "Fr", "FAIR\ndata", "ml", 51),
]
CAT = {"core": ("#14142b", "Core tools"), "scipy": (TEAL, "Scientific Python"),
       "viz": (AMBER, "Visualisation"), "chem": (VIOLET, "Chemistry & materials"),
       "ml": (ROSE, "Machine learning & AI")}


def library_table(name="library_table.png"):
    fig = plt.figure(figsize=(14, 5.6), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(-0.3, 18.3); ax.set_ylim(-1.7, 4.85); ax.axis("off"); ax.invert_yaxis()
    fig.patch.set_facecolor("white")
    for r, c, sym, nm, cat, ch in LIB_TILES:
        col = CAT[cat][0]
        ax.add_patch(FancyBboxPatch((c + 0.06, r + 0.06), 0.88, 0.88,
                                    boxstyle="round,pad=0,rounding_size=0.08", fc=col, ec="white", lw=1.5))
        ax.text(c + 0.12, r + 0.22, str(ch), color="white", fontsize=6.5, va="center", ha="left", alpha=0.85)
        ax.text(c + 0.5, r + 0.47, sym, color="white", fontsize=15, weight="bold", ha="center", va="center")
        ax.text(c + 0.5, r + 0.79, nm, color="white", fontsize=5.6, ha="center", va="center", linespacing=0.9)
    ax.text(9, -1.2, "Periodic Table of Python for Chemists", ha="center", va="center",
            fontsize=21, weight="bold", color=INK)
    ax.text(9, -0.55, "every tool and method used in Molecules, Data and Python\n(the small number on each tile is the chapter that introduces it)",
            ha="center", va="center", fontsize=9.5, color="#555", linespacing=1.3)
    for k, (cat, (col, lbl)) in enumerate(CAT.items()):
        x = 2.6 + k * 2.75
        ax.add_patch(FancyBboxPatch((x, 4.25), 0.32, 0.32, boxstyle="round,pad=0,rounding_size=0.05", fc=col, ec="none"))
        ax.text(x + 0.42, 4.41, lbl, fontsize=9, va="center", color=INK)
    fig.savefig(OUT / name, dpi=160, facecolor="white", bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def logo_with_title(name="logo_title.png"):
    """Sidebar logo: the hexagon mark above the book title."""
    fig = plt.figure(figsize=(3.2, 2.6), dpi=200)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 3.2); ax.set_ylim(0, 2.6); ax.axis("off")
    fig.patch.set_alpha(0)
    cx, cy, R = 1.6, 1.7, 0.62
    ax.add_patch(RegularPolygon((cx, cy), 6, radius=R, orientation=np.pi / 6, fill=False, ec=TEAL, lw=5))
    ax.add_patch(Circle((cx, cy), 0.36, fill=False, ec=VIOLET, lw=3))
    ang = np.pi / 6 + np.arange(6) * np.pi / 3
    for k, a in enumerate(ang):
        ax.add_patch(Circle((cx + R * np.cos(a), cy + R * np.sin(a)), 0.1, color=PALETTE[k % 4], zorder=5))
    xs = np.linspace(-0.28, 0.28, 40)
    ax.plot(cx + xs, cy + 0.22 * np.tanh(4 * xs / 0.62) - 0.01, color=AMBER, lw=3, solid_capstyle="round")
    ax.text(cx, 0.62, "Molecules, Data", ha="center", va="center", fontsize=13, weight="bold", color=VIOLET, style="italic")
    ax.text(cx, 0.28, "and Python", ha="center", va="center", fontsize=13, weight="bold", color=TEAL, style="italic")
    fig.savefig(OUT / name, transparent=True)
    plt.close(fig)


if __name__ == "__main__":
    library_table()
    logo_with_title()
    print("wrote library_table.png and logo_title.png")
