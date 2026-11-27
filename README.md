# Molecules, Data and Python

![Molecules, Data and Python](book/_static/social_card.png)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![Book](https://img.shields.io/badge/read-online-6c4ad6)](https://bahadursk.github.io/molecules-data-python)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

By **Dr. Bahadur Sk** ([ORCID 0000-0002-7098-5429](https://orcid.org/0000-0002-7098-5429)), Koneru Lakshmaiah Education Foundation.

An interactive, open Jupyter Book that teaches Python through chemistry, from school level to research level, and applies machine learning across chemistry and materials science.

- **Part I** Python fundamentals, every idea taught through a chemical problem
- **Part II** NumPy, pandas, Matplotlib, Seaborn, Plotly, SciPy, SymPy, pint
- **Part III** Cheminformatics with RDKit and 3D visualisation with py3Dmol
- **Part IV** Kinetics, thermodynamics, spectroscopy, quantum chemistry, MD, crystals
- **Part V** Machine-learning foundations for chemists
- **Part VI** ML in medicinal, organic, analytical, physical chemistry, catalysis, materials, energy, polymers, biochemistry, generative design and LLMs

## Build locally

```bash
conda env create -f environment.yml && conda activate mdp
python tools/build_book.py   # regenerate notebooks and the table of contents
jupyter-book build book      # HTML in book/_build/html
```

## Writing a chapter

Chapters are written as Jupytext MyST files in `src/<chapter>.md` and converted to notebooks in `book/` by `tools/build_book.py`. The outline lives in `tools/chapters.py`.

## Licence

Code: MIT. Text and figures: CC BY 4.0.

## How to cite

See [`CITATION.cff`](CITATION.cff), or use the "Cite this repository" button on GitHub. Each GitHub release is archived on Zenodo with its own DOI.

## Getting a DOI (one-time setup)

1. Author details are already in `CITATION.cff` and `.zenodo.json`.
2. Sign in to [zenodo.org](https://zenodo.org) with GitHub, open **Settings → GitHub**, and switch on this repository.
3. On GitHub, create a release (for example `v1.0.0`). Zenodo archives it and mints a DOI within minutes.
4. Replace `XXXXXXX` in the badge above and in `CITATION.cff` with the concept DOI Zenodo shows (it always points to the latest version).
