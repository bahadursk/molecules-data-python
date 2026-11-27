"""Generate the table of contents and chapter notebooks.

    python tools/build_book.py

* Chapters written in `src/<chapter>.md` (Jupytext MyST format) are converted
  to `book/<part>/<chapter>.ipynb`, which is what readers open and run.
* Chapters with no source yet get a "coming soon" starter notebook listing
  the planned sections, unless a notebook already exists.
"""
from pathlib import Path
import sys

import jupytext
import nbformat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from chapters import PARTS, APPENDICES, LEVEL_BADGE  # noqa: E402

BOOK = ROOT / "book"
SRC = ROOT / "src"
KERNEL = {"display_name": "Python 3", "language": "python", "name": "python3"}

# Official, freely available documentation for each library a chapter imports.
DOCS = {
    "numpy": ("NumPy documentation", "https://numpy.org/doc/stable/"),
    "pandas": ("pandas user guide", "https://pandas.pydata.org/docs/user_guide/"),
    "matplotlib": ("Matplotlib tutorials and gallery", "https://matplotlib.org/stable/tutorials/index.html"),
    "seaborn": ("seaborn tutorial", "https://seaborn.pydata.org/tutorial.html"),
    "plotly": ("Plotly Python graphing library", "https://plotly.com/python/"),
    "scipy": ("SciPy user guide", "https://docs.scipy.org/doc/scipy/tutorial/"),
    "sympy": ("SymPy tutorial", "https://docs.sympy.org/latest/tutorials/"),
    "pint": ("pint documentation", "https://pint.readthedocs.io/"),
    "uncertainties": ("uncertainties documentation", "https://uncertainties.readthedocs.io/"),
    "rdkit": ("The RDKit Getting Started guide", "https://www.rdkit.org/docs/GettingStartedInPython.html"),
    "py3Dmol": ("py3Dmol on PyPI", "https://pypi.org/project/py3Dmol/"),
    "pubchempy": ("PubChemPy documentation", "https://pubchempy.readthedocs.io/"),
    "chembl_webresource_client": ("ChEMBL web services", "https://chembl.gitbook.io/chembl-interface-documentation/web-services"),
    "sklearn": ("scikit-learn user guide", "https://scikit-learn.org/stable/user_guide.html"),
    "torch": ("PyTorch tutorials", "https://pytorch.org/tutorials/"),
    "pyscf": ("PySCF user guide", "https://pyscf.org/user.html"),
    "ase": ("ASE documentation", "https://wiki.fysik.dtu.dk/ase/"),
    "pymatgen": ("pymatgen documentation", "https://pymatgen.org/"),
    "mp_api": ("Materials Project API docs", "https://docs.materialsproject.org/downloading-data/using-the-api"),
    "MDAnalysis": ("MDAnalysis user guide", "https://userguide.mdanalysis.org/"),
}
PY_DOCS = ("The Python tutorial", "https://docs.python.org/3/tutorial/")
UNNUMBERED = {"exercises", "further reading", "summary"}


def number_headings(nb, chap_no):
    """Prefix the title with 'Chapter N:' and number level-2 sections N.1, N.2, ..."""
    import re
    sec = 0
    in_fence = False
    for cell in nb.cells:
        if cell.cell_type != "markdown":
            continue
        lines = cell.source.split("\n")
        for i, line in enumerate(lines):
            if line.lstrip().startswith(("```", "~~~")):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = re.match(r"^(#{1,2}) (.*)$", line)
            if not m:
                continue
            hashes, text = m.groups()
            if hashes == "#" and not text.startswith("Chapter "):
                lines[i] = f"# Chapter {chap_no}: {text}"
            elif hashes == "##" and text.strip().lower() not in UNNUMBERED and not re.match(r"^\d+\.\d+ ", text):
                sec += 1
                lines[i] = f"## {chap_no}.{sec} {text}"
        cell.source = "\n".join(lines)
        in_fence = False
    return nb


def further_reading(nb, chap_no):
    """Insert a Further Reading cell (official docs of the libraries used) before the exercises."""
    import re
    mods = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            for m in re.finditer(r"^(?:from|import) ([A-Za-z_][A-Za-z0-9_]*)", cell.source, re.M):
                if m.group(1) in DOCS and m.group(1) not in mods:
                    mods.append(m.group(1))
    items = [PY_DOCS] if chap_no <= 11 else []
    items += [DOCS[m] for m in mods]
    if not items:
        return nb
    md = "## Further Reading\n\n" + "\n".join(f"- [{t}]({u})" for t, u in items)
    md += "\n- Works cited in this chapter are listed in full in the [References](../appendices/A5_references.md)."
    cell = nbformat.v4.new_markdown_cell(md)
    for k, c in enumerate(nb.cells):
        m = re.search(r"^## (\d+\.\d+ )?Exercises", c.source, re.M) if c.cell_type == "markdown" else None
        if m:
            before, after = c.source[:m.start()].rstrip(), c.source[m.start():]
            if before:                      # split so Further Reading sits right before the heading
                c.source = after
                nb.cells.insert(k, nbformat.v4.new_markdown_cell(before))
                k += 1
            nb.cells.insert(k, cell)
            break
    else:
        nb.cells.append(cell)
    return nb


def stub(title, level, topics):
    bullets = "\n".join(f"- {t}" for t in topics)
    md = (
        f"# {title}\n\n"
        f"```{{admonition}} {LEVEL_BADGE[level]}\n:class: level-{level}\n"
        "This chapter is being written. The planned sections are listed below.\n```\n\n"
        f"## What you will learn\n\n{bullets}\n"
    )
    nb = nbformat.v4.new_notebook()
    nb.cells = [nbformat.v4.new_markdown_cell(md)]
    nb.metadata["kernelspec"] = KERNEL
    return nb


def main():
    toc = ["format: jb-book", "root: intro", "parts:",
           "  - caption: Getting Started", "    chapters:",
           "      - file: front/preface", "      - file: front/how_to_use", "      - file: front/installation"]
    for part in PARTS:
        toc += [f'  - caption: "{part["caption"]}"', "    chapters:"]
        (BOOK / part["dir"]).mkdir(parents=True, exist_ok=True)
        for name, title, level, topics in part["chapters"]:
            out = BOOK / part["dir"] / f"{name}.ipynb"
            src = SRC / f"{name}.md"
            if src.exists():
                nb = jupytext.read(src, fmt="md:myst")
                chap_no = int(name.split("_")[0])
                nb = further_reading(number_headings(nb, chap_no), chap_no)
                nb.metadata["kernelspec"] = KERNEL
                nb.metadata.pop("jupytext", None)
                nbformat.write(nb, out)
                print("built  ", out.relative_to(ROOT))
            elif not out.exists():
                nbformat.write(stub(title, level, topics), out)
                print("stubbed", out.relative_to(ROOT))
            toc.append(f"      - file: {part['dir']}/{name}")
    toc += ["  - caption: Appendices", "    chapters:"]
    (BOOK / "appendices").mkdir(exist_ok=True)
    for name, title in APPENDICES:
        p = BOOK / "appendices" / f"{name}.md"
        if not p.exists():
            p.write_text(f"# {title}\n\n*Coming soon.*\n")
        toc.append(f"      - file: appendices/{name}")
    (BOOK / "_toc.yml").write_text("\n".join(toc) + "\n")
    print("wrote   book/_toc.yml")


if __name__ == "__main__":
    main()
