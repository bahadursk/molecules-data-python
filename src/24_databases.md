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

Millions of structures, properties and assay results are freely available: **PubChem** {cite:p}`kim2023`, **ChEMBL** {cite:p}`mendez2019`, the **Protein Data Bank** {cite:p}`berman2000` and the **Materials Project** {cite:p}`jain2013`. All of them offer web APIs that Python can query. PubChem holds over a hundred million compounds with computed and measured properties; ChEMBL collects measured activities of drug-like molecules against biological targets, extracted from the medicinal chemistry literature; the PDB stores experimentally determined 3D structures of proteins and nucleic acids; and the Materials Project provides computed properties of inorganic crystals. Querying them from code instead of a web browser means you can download thousands of records reproducibly, clean them with pandas and feed them straight into analysis or machine learning. This chapter explains how web APIs work, shows the main Python clients for each database, and ends with the habits that keep your use of public data responsible.

```{admonition} Running these examples
:class: tip
Code that contacts a web service is shown in plain grey blocks: copy it into Colab or Jupyter to run it. The book itself is built offline, so the executed examples below work with a small saved response instead.
```

## How web APIs work

An **API** (application programming interface) is a defined way for one program to ask another for something. A **web API**, often called a **REST API**, is a set of web addresses that return data, usually as JSON (Chapter 8), instead of a page meant for people. The address itself encodes the question. In PubChem's PUG-REST API, for instance, the address is built from parts that read like a sentence: which database (`compound`), how to find the record (`name/caffeine`), what to return (`property/MolecularFormula,...`) and in what format (`JSON`). You build the address, send a request with the `requests` library, check that the request succeeded, and parse the reply.

Building the address is just string formatting, so it is easy to change the compound or the properties:

```{code-cell} ipython3
base = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
name, props_wanted = "caffeine", ["MolecularFormula", "MolecularWeight"]
url = f"{base}/compound/name/{name}/property/{','.join(props_wanted)}/JSON"
print(url)
```

Sending the request takes three lines. `timeout=30` stops the program waiting for ever if the server does not answer, and `raise_for_status()` turns an HTTP error (such as 404, "not found") into a Python exception (Chapter 9):

```python
import requests
url = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/caffeine/property/MolecularFormula,MolecularWeight,XLogP,CanonicalSMILES/JSON"
reply = requests.get(url, timeout=30)
reply.raise_for_status()
data = reply.json()
```

The reply has this shape (saved here so the book runs offline). It is nested dictionaries and lists, so you reach the properties by indexing step by step:

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

Writing addresses by hand is fine for a single query, but a **client library** is more convenient: it builds the addresses, sends the requests and turns the replies into Python objects for you. The `pubchempy` package wraps the PubChem API in friendly Python objects. A `Compound` object has attributes such as `cid` (PubChem's compound identifier), `molecular_formula` and `isomeric_smiles`. `get_properties` fetches several properties for several compounds at once, straight into a DataFrame, and a similarity search returns compounds whose fingerprints are close to a query (Chapter 21).

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

ChEMBL collects measured activities of molecules against biological targets: how strongly each compound inhibits an enzyme or binds a receptor, with the assay details and the paper it came from. It is the main public source of training data for drug-discovery models (Chapter 36). Its official client, `chembl_webresource_client`, lets you filter records much as you would filter a DataFrame: first find the target, then the activities measured against it, restricted to one activity type and unit so that the values are comparable.

The most common activity measure is the **IC₅₀**, the concentration of an inhibitor that halves the activity of its target; a smaller IC₅₀ means a more potent compound. IC₅₀ values span many orders of magnitude, so they are usually converted to the logarithmic **pIC₅₀**, just as concentrations of H⁺ become pH:

```{code-cell} ipython3
import numpy as np
for ic50_nM in [1, 10, 1000]:
    print(f"IC50 = {ic50_nM:5} nM  ->  pIC50 = {-np.log10(ic50_nM * 1e-9):.1f}")
```

Each factor of ten in potency adds one pIC₅₀ unit, and higher is better.

```python
from chembl_webresource_client.new_client import new_client
target = new_client.target.filter(target_synonym__icontains="acetylcholinesterase", organism="Homo sapiens")[0]
acts = new_client.activity.filter(target_chembl_id=target["target_chembl_id"], standard_type="IC50",
                                  standard_units="nM").only(["molecule_chembl_id", "canonical_smiles", "standard_value"])
import pandas as pd
df = pd.DataFrame(list(acts[:500]))
```

After download, a typical clean-up converts IC₅₀ to pIC₅₀ = −log₁₀(IC₅₀ / M) and removes duplicates. The same compound is often measured several times, so its values are combined, usually by taking the median, which is less affected by one outlying measurement than the mean. Here it is on a small illustrative table of acetylcholinesterase inhibitors used to treat Alzheimer's disease:

```{code-cell} ipython3
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

The RCSB PDB offers two kinds of service: a data API that returns information about an entry as JSON (its title, experimental method and resolution, where a smaller number in ångströms means a sharper structure), and a file server that returns the coordinates themselves in PDB or mmCIF format. The coordinates can be passed to py3Dmol (Chapter 23) for viewing.

```python
import requests
entry = requests.get("https://data.rcsb.org/rest/v1/core/entry/4EY7", timeout=30).json()
print(entry["struct"]["title"], entry["rcsb_entry_info"]["resolution_combined"])
pdb_text = requests.get("https://files.rcsb.org/download/4EY7.pdb", timeout=30).text
```

## The Materials Project

The Materials Project computes the properties of inorganic crystals with density functional theory (Chapter 28): band gaps, formation energies, elastic constants and more. Its `mp_api` client searches by elements and by ranges of properties. Access requires a free **API key**, a personal password-like string that identifies you to the service. Keep keys out of notebooks you share; store them in an environment variable instead. `energy_above_hull` measures thermodynamic stability: zero means the compound is stable, and larger values mean it would tend to decompose.

```python
from mp_api.client import MPRester
with MPRester("YOUR_API_KEY") as mpr:
    docs = mpr.materials.summary.search(elements=["Li", "Fe", "O"], band_gap=(0.5, 4),
                                        fields=["material_id", "formula_pretty", "band_gap", "energy_above_hull"])
```

## Using public data responsibly

Public databases are shared resources maintained by others, and using them well is part of good scientific practice. Four habits matter:

1. **Check the licence.** PubChem data is public domain; ChEMBL is CC BY-SA 3.0; PDB entries are CC0. Credit sources and respect share-alike terms when you redistribute data.
2. **Be polite.** Respect rate limits (PubChem asks for at most 5 requests per second), add `time.sleep()` between calls and cache results to disk.
3. **Record provenance.** Save the query, the date and the database release with your data.
4. **Cite the database** in your publications.

A **cache** stores each reply on disk the first time you download it, so that re-running a notebook reads the saved copy instead of asking the server again. It makes your analysis faster, keeps working offline, reduces load on the database, and freezes the data you actually used. The function below names each cached file after a *hash* of its address (a short fixed-length code computed from the text), so every distinct query gets its own file.

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

```{admonition} Summary
:class: tip
- A REST API is a web address that returns data (usually JSON); build the URL, `requests.get(url, timeout=...)`, `raise_for_status()`, then `.json()`.
- Client libraries (`pubchempy`, `chembl_webresource_client`, `mp_api`) wrap APIs in Python objects.
- Convert IC₅₀ to pIC₅₀ = −log₁₀(IC₅₀/M) and combine duplicates with the median.
- Check licences, respect rate limits, cache replies, record provenance and cite the databases.
```

## Exercises

**★ 1.** Using PubChemPy, download the molecular weight and XLogP of ten common solvents and plot them.

**★★ 2.** From ChEMBL, download IC₅₀ values for a target of your choice, convert to pIC₅₀ and draw the ten most potent molecules.

**★★★ 3.** Write a function that fetches a PDB entry, extracts the ligand records (`HETATM`) and displays the binding site with py3Dmol.
