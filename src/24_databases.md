---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Chemical Databases from Python

```{admonition} 🟡 Intermediate · undergraduate
:class: level-2
Assumes Chapters 8, 15 and 19–20.
```

Millions of structures, properties and assay results are freely available: **PubChem** {cite:p}`kim2023`, **ChEMBL** {cite:p}`mendez2019`, the **Protein Data Bank** {cite:p}`berman2000` and the **Materials Project** {cite:p}`jain2013`. All of them offer web APIs that Python can query.

```{admonition} Running these examples
:class: tip
Code that contacts a web service is shown in plain grey blocks: copy it into Colab or Jupyter to run it. The book itself is built offline, so the executed examples below work with a small saved response instead.
```

## How web APIs work

A **REST API** is a web address that returns data (usually JSON) instead of a web page. You build the address, request it, and parse the reply.

```python
import requests
url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/caffeine/property/MolecularFormula,MolecularWeight,XLogP,CanonicalSMILES/JSON"
reply = requests.get(url, timeout=30)
reply.raise_for_status()
data = reply.json()
```

The reply has this shape (saved here so the book runs offline):

```{code-cell} ipython3
import json
reply_text = """{"PropertyTable": {"Properties": [{"CID": 2519, "MolecularFormula": "C8H10N4O2",
  "MolecularWeight": "194.19", "CanonicalSMILES": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "XLogP": -0.1}]}}"""
data = json.loads(reply_text)
props = data["PropertyTable"]["Properties"][0]
props
```

```{code-cell} ipython3
from rdkit import Chem
from rdkit.Chem import Draw
Draw.MolToImage(Chem.MolFromSmiles(props["CanonicalSMILES"]), size=(300, 200), legend=f"CID {props['CID']}")
```

## PubChem with PubChemPy

The `pubchempy` package wraps the PubChem API in friendly Python objects.

```python
import pubchempy as pcp
c = pcp.get_compounds("ibuprofen", "name")[0]
print(c.cid, c.molecular_formula, c.molecular_weight, c.isomeric_smiles)

# Many compounds at once into a pandas DataFrame
df = pcp.get_properties(["MolecularWeight", "XLogP", "TPSA"], ["aspirin", "caffeine", "nicotine"], "name", as_dataframe=True)

# Similarity search: compounds ≥ 95 % similar to aspirin
similar = pcp.get_compounds("CC(=O)Oc1ccccc1C(=O)O", "smiles", searchtype="similarity", Threshold=95, listkey_count=20)
```

## ChEMBL: bioactivity data

ChEMBL collects measured activities of molecules against biological targets.

```python
from chembl_webresource_client.new_client import new_client
target = new_client.target.filter(target_synonym__icontains="acetylcholinesterase", organism="Homo sapiens")[0]
acts = new_client.activity.filter(target_chembl_id=target["target_chembl_id"], standard_type="IC50",
                                  standard_units="nM").only(["molecule_chembl_id", "canonical_smiles", "standard_value"])
import pandas as pd
df = pd.DataFrame(list(acts[:500]))
```

After download, a typical clean-up converts IC₅₀ to pIC₅₀ = −log₁₀(IC₅₀ / M) and removes duplicates. Here it is on a small illustrative table:

```{code-cell} ipython3
import numpy as np
import pandas as pd
df = pd.DataFrame({
    "molecule": ["donepezil", "galantamine", "rivastigmine", "tacrine", "huperzine A", "donepezil"],
    "smiles": ["COc1cc2CC(CC3CCN(Cc4ccccc4)CC3)C(=O)c2cc1OC", "COc1ccc2C[N@@](C)CC[C@@]34C=C[C@H](O)C[C@@H]3Oc1c24",
               "CCN(C)C(=O)Oc1cccc(c1)[C@H](C)N(C)C", "Nc1c2CCCCc2nc2ccccc12",
               "C/C=C1/[C@@H]2Cc3[nH]c(=O)ccc3[C@]1(N)CC(C)=C2", "COc1cc2CC(CC3CCN(Cc4ccccc4)CC3)C(=O)c2cc1OC"],
    "IC50_nM": [6.7, 410, 4300, 77, 82, 8.1],
})
df["pIC50"] = -np.log10(df["IC50_nM"] * 1e-9)
clean = df.groupby(["molecule", "smiles"], as_index=False)["pIC50"].median().sort_values("pIC50", ascending=False)
clean
```

```{code-cell} ipython3
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in clean.smiles],
                     legends=[f"{m}  pIC50 {p:.1f}" for m, p in zip(clean.molecule, clean.pIC50)],
                     molsPerRow=5, subImgSize=(200, 170))
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Activity values from different labs and assays are not always comparable. Before modelling, check the assay type, units and relation (`=`, `<`, `>`), and keep track of where each number came from.
```

## The Protein Data Bank

```python
import requests
entry = requests.get("https://data.rcsb.org/rest/v1/core/entry/4EY7", timeout=30).json()
print(entry["struct"]["title"], entry["rcsb_entry_info"]["resolution_combined"])
pdb_text = requests.get("https://files.rcsb.org/download/4EY7.pdb", timeout=30).text
```

## The Materials Project

Requires a free API key from materialsproject.org.

```python
from mp_api.client import MPRester
with MPRester("YOUR_API_KEY") as mpr:
    docs = mpr.materials.summary.search(elements=["Li", "Fe", "O"], band_gap=(0.5, 4),
                                        fields=["material_id", "formula_pretty", "band_gap", "energy_above_hull"])
```

## Using public data responsibly

1. **Check the licence.** PubChem data is public domain; ChEMBL is CC BY-SA 3.0; PDB entries are CC0. Credit sources and respect share-alike terms when you redistribute data.
2. **Be polite.** Respect rate limits (PubChem asks for at most 5 requests per second), add `time.sleep()` between calls and cache results to disk.
3. **Record provenance.** Save the query, the date and the database release with your data.
4. **Cite the database** in your publications.

```{code-cell} ipython3
import time, hashlib
from pathlib import Path

CACHE = Path("api_cache"); CACHE.mkdir(exist_ok=True)

def cached_get(url, fetch):
    """Return cached text for url, calling fetch(url) only on a cache miss."""
    key = CACHE / (hashlib.sha1(url.encode()).hexdigest() + ".json")
    if key.exists():
        return key.read_text()
    text = fetch(url)
    key.write_text(text)
    time.sleep(0.25)
    return text

# Demonstration with a fake fetcher (use requests.get(url).text in real life)
print(cached_get("https://example.org/caffeine", lambda u: reply_text)[:60], "...")
```

## Exercises

**★ 1.** Using PubChemPy, download the molecular weight and XLogP of ten common solvents and plot them.

**★★ 2.** From ChEMBL, download IC₅₀ values for a target of your choice, convert to pIC₅₀ and draw the ten most potent molecules.

**★★★ 3.** Write a function that fetches a PDB entry, extracts the ligand records (`HETATM`) and displays the binding site with py3Dmol.
