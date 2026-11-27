---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Large Language Models and AI Agents in Chemistry

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 32, 34 and 45.
```

Language models read and write text, and much of chemistry *is* text: SMILES, reaction equations, procedures, papers. Models fine-tuned on chemical text can predict properties from a handful of examples {cite:p}`jablonka2024`, word embeddings trained on the literature encode latent materials knowledge {cite:p}`tshitoyan2019`, and language models wired to real tools can plan and execute experiments {cite:p}`bran2024,boiko2023`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path
from collections import Counter
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, rdMolDescriptors, Draw
rng = np.random.default_rng(0)

DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name, **kw):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name, **kw)
df = load_data("molecules_sim.csv")
```

## Chemistry as a language

A tokeniser splits text into units a model can handle. For chemistry, the unit matters: characters, SMILES atoms, or learned subwords.

```{code-cell} ipython3
SMILES_ATOM = re.compile(r"(\[[^\]]+]|Br|Cl|Si|@@|%\d{2}|[BCNOPSFIbcnops]|\(|\)|\.|=|#|-|\+|\\|/|:|~|@|\*|\$|\d)")
s = "CC(=O)Oc1ccccc1C(=O)O"
print("characters :", list(s)[:14], "...")
print("atom tokens:", SMILES_ATOM.findall(s))
```

### Byte-pair encoding, from scratch

BPE repeatedly merges the most frequent adjacent pair, which is how modern tokenisers learn chemical motifs such as `C(=O)O` as single units.

```{code-cell} ipython3
def learn_bpe(corpus, n_merges=60):
    seqs = [SMILES_ATOM.findall(s) for s in corpus]
    merges = []
    for _ in range(n_merges):
        pairs = Counter()
        for seq in seqs:
            pairs.update(zip(seq, seq[1:]))
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append((best, pairs[best]))
        joined = "".join(best)
        new = []
        for seq in seqs:
            out, i = [], 0
            while i < len(seq):
                if i + 1 < len(seq) and (seq[i], seq[i + 1]) == best:
                    out.append(joined); i += 2
                else:
                    out.append(seq[i]); i += 1
            new.append(out)
        seqs = new
    return merges

merges = learn_bpe(df.smiles.tolist(), 60)
print("Learned chemical 'words':")
for (pair, count) in merges[:18]:
    print(f"  {''.join(pair):12s} seen {count} times")
```

The model discovers carbonyls, aromatic rings and common substituents without being told any chemistry.

## Word embeddings capture chemical meaning

Tshitoyan and co-workers showed that embeddings trained on materials abstracts place related concepts near each other, and can even rank candidate materials before anyone studied them {cite:p}`tshitoyan2019`. Here is the mechanism on a small synthetic corpus, with embeddings learned by factorising a co-occurrence matrix.

```{code-cell} ipython3
SENTENCES = [
    "lithium cobalt oxide cathode battery high capacity",
    "lithium iron phosphate cathode battery safe long cycle life",
    "sodium iron phosphate cathode battery cheap sodium ion",
    "graphite anode battery lithium intercalation",
    "silicon anode battery high capacity volume expansion",
    "perovskite solar cell lead iodide high efficiency absorber",
    "silicon solar cell wafer efficiency absorber",
    "cadmium telluride solar cell thin film absorber",
    "titanium dioxide photocatalyst water splitting band gap",
    "zinc oxide photocatalyst wide band gap semiconductor",
    "palladium catalyst cross coupling suzuki reaction",
    "nickel catalyst cross coupling cheap earth abundant",
    "platinum catalyst fuel cell oxygen reduction",
    "zeolite catalyst cracking porous acid site",
    "metal organic framework porous gas storage methane",
    "activated carbon porous gas storage cheap",
    "copper catalyst click reaction azide alkyne",
    "iron catalyst earth abundant oxidation cheap",
    "lead free perovskite tin iodide solar absorber",
    "solid electrolyte battery sulfide lithium conductivity",
]

words = sorted({w for s in SENTENCES for w in s.split()})
w2i = {w: i for i, w in enumerate(words)}
C = np.zeros((len(words), len(words)))
for s in SENTENCES:
    toks = s.split()
    for i, a in enumerate(toks):
        for b in toks[max(0, i - 4):i + 5]:
            if a != b:
                C[w2i[a], w2i[b]] += 1
M = np.log1p(C)
U, S, _ = np.linalg.svd(M - M.mean(0))
E = U[:, :8] * S[:8]                                    # 8-dimensional embeddings
E = E / np.linalg.norm(E, axis=1, keepdims=True)

def nearest(word, k=5):
    sims = E @ E[w2i[word]]
    order = np.argsort(-sims)
    return [(words[i], round(float(sims[i]), 2)) for i in order if words[i] != word][:k]

for w in ["cathode", "absorber", "catalyst", "porous"]:
    print(f"{w:10s} → {nearest(w)}")
```

```{code-cell} ipython3
from sklearn.decomposition import PCA
P = PCA(2).fit_transform(E)
fig, ax = plt.subplots(figsize=(9, 7))
GROUPS = {"battery": "#e5487a", "solar": "#f2a93b", "catalyst": "#0f9d8a", "porous": "#6c4ad6"}
for w, (x, y) in zip(words, P):
    colour = "grey"
    for key, c in GROUPS.items():
        if max(E[w2i[w]] @ E[w2i[key]], 0) > 0.35:
            colour = c
    ax.scatter(x, y, s=20, color=colour)
    ax.annotate(w, (x, y), fontsize=8, color=colour, textcoords="offset points", xytext=(4, 3))
ax.set(title="Embedding space groups applications without being told", xticks=[], yticks=[])
plt.show()
```

## Text mining a procedure

Much chemical knowledge sits in prose. Rule-based extraction gets a long way, and is a good baseline before reaching for a language model.

```{code-cell} ipython3
PROCEDURE = """To a solution of 4-bromoanisole (1.87 g, 10.0 mmol) in dry toluene (40 mL) was added
phenylboronic acid (1.46 g, 12.0 mmol), potassium carbonate (2.76 g, 20.0 mmol) and
Pd(PPh3)4 (231 mg, 0.20 mmol, 2 mol%). The mixture was heated at 110 °C for 16 h under argon.
After cooling to room temperature, the mixture was diluted with ethyl acetate (50 mL), washed with
water (2 x 30 mL) and brine (30 mL), dried over Na2SO4 and concentrated. Purification by column
chromatography (hexane/EtOAc 95:5) gave 4-methoxybiphenyl as a white solid (1.62 g, 88%)."""

AMOUNT = re.compile(r"\(\s*(\d+\.?\d*)\s?(mg|g|mL|L)\s*(?:,\s*(\d+\.?\d*)\s?(mmol|mol))?[^)]*\)")
STOP = {"to", "a", "solution", "of", "in", "dry", "was", "added", "and", "with", "the", "mixture",
        "diluted", "washed", "gave", "over", "by", "as", "white", "solid", "after", "cooling", "room",
        "temperature", "purification", "column", "chromatography"}

def preceding_name(text, start, max_words=4):
    """Take the words just before a parenthesis, dropping prose words."""
    before = text[:start].replace("\n", " ")
    before = before[before.rfind(")") + 1:]            # never read back past a previous amount
    words = re.findall(r"[\w\-\(\)\d,']+", before)
    name = []
    for w in reversed(words[-max_words:]):
        if w.lower() in STOP:
            break
        name.insert(0, w)
    return " ".join(name).strip(" ,")

print("Reagents and amounts")
for match in AMOUNT.finditer(PROCEDURE):
    amount, unit, mol, molunit = match.groups()
    name = preceding_name(PROCEDURE, match.start())
    if name:
        print(f"  {name:22s} {amount:>6s} {unit:3s} " + (f"{mol} {molunit}" if mol else ""))

print("\nNote the two mistakes: the catalyst Pd(PPh3)4 is cut at its own bracket, and one name keeps a\n"
      "leading fragment. Rule-based extraction is a baseline, not a solution.\n")

print("Conditions")
for label, pattern in [("temperature", r"(\d+)\s?°C"), ("time", r"(\d+\.?\d*)\s?(?:h|hours|min)"),
                       ("atmosphere", r"under (argon|nitrogen|air|vacuum)"), ("yield", r"(\d+)\s?%\)")]:
    print(f"  {label:12s} {re.findall(pattern, PROCEDURE)}")
```

```{code-cell} ipython3
mentions = {"4-bromoanisole": "COc1ccc(Br)cc1", "phenylboronic acid": "OB(O)c1ccccc1",
            "4-methoxybiphenyl": "COc1ccc(-c2ccccc2)cc1"}
Draw.MolsToGridImage([Chem.MolFromSmiles(s) for s in mentions.values()], legends=list(mentions),
                     molsPerRow=3, subImgSize=(230, 170))
```

This works, but notice how much of the code is bookkeeping about prose rather than chemistry, and how easily a slightly different sentence would break it. That brittleness is why modern pipelines hand the paragraph to a language model and ask for structured output (a JSON list of reagents, amounts and conditions), then validate every field with exactly the kind of checks below.

## In-context learning: predicting from a few examples

A language model can be given examples in its prompt and asked to continue the pattern, with no training at all. Whether this beats a small fitted model is an empirical question, and often it does not {cite:p}`jablonka2024`. We can measure the idea honestly with a nearest-neighbour "prompt" baseline: the same information the model would see.

```{code-cell} ipython3
from rdkit.Chem import rdFingerprintGenerator, DataStructs
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
mols = [Chem.MolFromSmiles(s) for s in df.smiles]
fps = [gen.GetFingerprint(m) for m in mols]
X = np.array([gen.GetCountFingerprintAsNumPy(m) for m in mols])
y = df.logS_sim.values

idx = rng.permutation(len(df))
test = idx[:300]
results = []
for n_shot in [5, 10, 20, 50, 100, 400, 1500]:
    train = idx[300:300 + n_shot]
    # "In-context": answer with the mean of the k most similar examples in the prompt
    preds_icl = []
    for t in test:
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fps[t], [fps[i] for i in train]))
        k = min(3, len(train))
        top = np.argsort(-sims)[:k]
        w = sims[top] + 1e-6
        preds_icl.append(float(np.average(y[train][top], weights=w)))
    rf = RandomForestRegressor(200, n_jobs=-1, random_state=0).fit(X[train], y[train])
    results.append({"examples": n_shot, "similarity prompt (ICL-like)": mean_absolute_error(y[test], preds_icl),
                    "fitted random forest": mean_absolute_error(y[test], rf.predict(X[test]))})
res = pd.DataFrame(results).set_index("examples")
res.plot(marker="o", figsize=(7, 4.2), color=["#e5487a", "#0f9d8a"])
plt.ylabel("MAE on logS"); plt.xscale("log"); plt.title("Few examples favour retrieval; many favour fitting")
plt.show()
res.round(3)
```

With a handful of examples, reasoning by analogy from similar molecules wins; once there are hundreds, a fitted model is better. That is the practical rule for deciding whether to prompt or to train.

## A tool-using agent, written out explicitly

An "agent" is a loop: the model decides which tool to call, the tool runs, the result goes back into the context. The tools are ordinary Python functions. Here is the loop with a rule-based planner standing in for the model, so you can see the mechanism with nothing hidden.

```{code-cell} ipython3
def tool_lookup(name):
    """Resolve a trivial name to SMILES (a real agent would query PubChem)."""
    table = {"aspirin": "CC(=O)Oc1ccccc1C(=O)O", "caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",
             "paracetamol": "CC(=O)Nc1ccc(O)cc1", "ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O"}
    return table.get(name.lower(), f"ERROR: {name} not found")

def tool_properties(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return "ERROR: invalid SMILES"
    return {"formula": rdMolDescriptors.CalcMolFormula(m), "MW": round(Descriptors.MolWt(m), 2),
            "logP": round(Crippen.MolLogP(m), 2), "TPSA": round(rdMolDescriptors.CalcTPSA(m), 1),
            "HBD": rdMolDescriptors.CalcNumHBD(m)}

def tool_predict_solubility(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        return "ERROR: invalid SMILES"
    rf = RandomForestRegressor(200, n_jobs=-1, random_state=0).fit(X, y)
    return round(float(rf.predict(gen.GetCountFingerprintAsNumPy(m)[None])[0]), 2)

def tool_substructure(smiles, smarts):
    m, p = Chem.MolFromSmiles(smiles), Chem.MolFromSmarts(smarts)
    if m is None or p is None:
        return "ERROR: invalid input"
    return len(m.GetSubstructMatches(p))

TOOLS = {"lookup": tool_lookup, "properties": tool_properties, "solubility": tool_predict_solubility,
         "substructure": tool_substructure}

def planner(task, scratchpad):
    """Stands in for a language model: decides the next tool call from the task and what is known."""
    if "smiles" not in scratchpad:
        return ("lookup", (task["compound"],))
    if "props" not in scratchpad:
        return ("properties", (scratchpad["smiles"],))
    if "logS" not in scratchpad:
        return ("solubility", (scratchpad["smiles"],))
    if "acid" not in scratchpad:
        return ("substructure", (scratchpad["smiles"], "[CX3](=O)[OX2H1]"))
    return None

def run_agent(task, max_steps=8, verbose=True):
    scratchpad, trace = {}, []
    keys = ["smiles", "props", "logS", "acid"]
    for step in range(max_steps):
        decision = planner(task, scratchpad)
        if decision is None:
            break
        tool, args = decision
        result = TOOLS[tool](*args)
        trace.append((tool, args, result))
        if verbose:
            print(f"step {step + 1}: {tool}{args} → {result}")
        if isinstance(result, str) and result.startswith("ERROR"):
            return {"error": result, "trace": trace}
        scratchpad[keys[len(scratchpad)]] = result
    return {"answer": scratchpad, "trace": trace}

out = run_agent({"compound": "aspirin"})
```

```{code-cell} ipython3
summary = out["answer"]
print(f"\nAspirin ({summary['smiles']}) has formula {summary['props']['formula']}, MW {summary['props']['MW']},")
print(f"predicted logS {summary['logS']}, and contains {summary['acid']} carboxylic acid group(s).")
```

The agent's answer is only as trustworthy as its tools, and every claim it makes is traceable to one of them. That traceability is the whole point: it is what separates a useful scientific assistant from a confident guess.

## Where models go wrong

```{code-cell} ipython3
BAD_OUTPUTS = {"C1CC(C": "unbalanced ring and bracket", "c1ccccc1C(C)(C)(C)C": "five-valent carbon",
               "CC(=O)O)C": "stray closing bracket", "N(C)(C)(C)(C)C": "five bonds to nitrogen",
               "CC(=O)Oc1ccccc1C(=O)O": "actually fine"}
for smi, note in BAD_OUTPUTS.items():
    m = Chem.MolFromSmiles(smi)
    print(f"{smi:26s} {'valid  ' if m else 'INVALID'} ({note})")
```

```{admonition} ⚗️ Chemistry note
:class: chem-note
Any structure a language model writes must be parsed and sanitised before use. RDKit returning `None` is your friend: it is the difference between a caught error and a fabricated molecule in a results table.
```

## Responsible use

```{admonition} A checklist before you trust or publish model output
:class: warning
1. **Validate** every structure, number and unit with code, as above.
2. **Cite the source** of any fact the model asserts, or verify it yourself. A model's fluency is not evidence.
3. **Record the prompt, the model version and the date.** Models change; results must stay reproducible.
4. **Disclose AI assistance** as your journal, institution or funder requires.
5. **Keep a human responsible.** The author of a paper is accountable for every claim in it.
6. **Think about misuse.** The same generative tools that design medicines can design toxins {cite:p}`urbina2022`. Do not build or publish systems aimed at harmful molecules, and treat screening against hazard lists as a basic professional duty.
```

## Exercises

**★ 1.** Learn BPE merges on only the aromatic molecules in the dataset. Which "words" appear that did not before?

**★★ 2.** Add a `tool_similar_molecules` function to the agent and extend the planner to use it.

**★★★ 3.** Replace the rule-based planner with a real language model through an API, keeping the same tools, and compare the traces. Count how often it calls a tool with invalid arguments.
