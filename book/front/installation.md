# Installing Python and the Libraries

## Option 1: no installation (recommended to start)

Click the 🚀 rocket icon at the top of any chapter and choose **Colab** or **Binder**. In Colab, run this in the first cell:

```ipython
!pip install rdkit py3Dmol pint uncertainties
```

## Option 2: on your own computer

1. Install [Miniforge](https://github.com/conda-forge/miniforge), a small conda distribution.
2. Download this book's repository and, in a terminal inside it, run:

```bash
conda env create -f environment.yml
conda activate mdp
jupyter lab
```

3. Open any notebook in `book/` and run the cells with **Shift + Enter**.

## Checking your setup

```python
import numpy, pandas, matplotlib, rdkit
print("RDKit", rdkit.__version__, "is ready")
```
