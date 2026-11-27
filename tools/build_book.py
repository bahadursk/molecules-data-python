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
