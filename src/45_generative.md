---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Generative Models and Inverse Design

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 32, 34 and 35.
```

Normal modelling goes structure → property. **Inverse design** goes property → structure: tell me a molecule with this band gap, this solubility, this activity. Variational autoencoders over SMILES {cite:p}`gomezbombarelli2018`, robust string grammars {cite:p}`krenn2020`, reinforcement learning and diffusion models all attack this problem {cite:p}`sanchezlengeling2018`.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from pathlib import Path
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import Draw, Descriptors, Crippen, QED, rdFingerprintGenerator
RDLogger.DisableLog("rdApp.*")
torch.manual_seed(0)
rng = np.random.default_rng(0)

DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name, **kw):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name, **kw)
df = load_data("molecules_sim.csv")
```

## Baseline 1: generate by enumeration

The honest baseline for any generative model is brute-force enumeration of a chemically sensible space.

```{code-cell} ipython3
import itertools
SCAFFOLDS = ["c1ccc({R})cc1", "c1cc({R})ncc1", "C1CCN({R})CC1", "c1ccc2[nH]c({R})cc2c1"]
GROUPS = ["C", "CC", "O", "OC", "N", "NC", "F", "Cl", "C#N", "C(=O)O", "C(=O)N", "S(=O)(=O)C"]
enumerated = []
for scaf, g in itertools.product(SCAFFOLDS, GROUPS):
    m = Chem.MolFromSmiles(scaf.replace("{R}", g))
    if m:
        enumerated.append(Chem.MolToSmiles(m))
print(len(enumerated), "molecules by enumeration; all valid by construction")
```

Enumeration is exhaustive but rigid: it can only produce what its rules allow. Generative models learn the rules from data.

## A character-level SMILES generator

The simplest learned generator is a language model: predict the next character, then sample.

```{code-cell} ipython3
smiles = [s for s in df.smiles if len(s) <= 40]
chars = sorted({c for s in smiles for c in s})
stoi = {c: i + 2 for i, c in enumerate(chars)}      # 0 = pad, 1 = start/end
itos = {i: c for c, i in stoi.items()}
L = max(len(s) for s in smiles) + 2

def encode(s):
    ids = [1] + [stoi[c] for c in s] + [1]
    return ids + [0] * (L - len(ids))

data = torch.tensor([encode(s) for s in smiles])
print(len(smiles), "training SMILES,", len(chars), "characters, max length", L)

class CharRNN(nn.Module):
    def __init__(self, n_vocab, d=128, layers=2):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, d, padding_idx=0)
        self.rnn = nn.GRU(d, d, layers, batch_first=True, dropout=0.1)
        self.out = nn.Linear(d, n_vocab)
    def forward(self, x, h=None):
        z, h = self.rnn(self.emb(x), h)
        return self.out(z), h

model = CharRNN(len(chars) + 2)
opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
loss_fn = nn.CrossEntropyLoss(ignore_index=0)
losses = []
for epoch in range(30):
    perm = torch.randperm(len(data))
    tot = 0.0
    for i in range(0, len(data), 128):
        b = data[perm[i:i + 128]]
        logits, _ = model(b[:, :-1])
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(b)
    losses.append(tot / len(data))
plt.plot(losses, color="#6c4ad6"); plt.xlabel("epoch"); plt.ylabel("cross entropy")
plt.title("Learning the grammar of SMILES"); plt.show()
```

```{code-cell} ipython3
@torch.no_grad()
def sample(n=400, temperature=1.0, max_len=L):
    model.eval()
    x = torch.ones(n, 1, dtype=torch.long)
    h = None
    done = torch.zeros(n, dtype=torch.bool)
    out = [[] for _ in range(n)]
    for _ in range(max_len):
        logits, h = model(x, h)
        probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
        probs[:, 0] = 0                                     # never sample padding
        nxt = torch.multinomial(probs, 1)
        for k in range(n):
            if not done[k]:
                if nxt[k, 0].item() == 1:
                    done[k] = True
                else:
                    out[k].append(itos[nxt[k, 0].item()])
        x = nxt
        if done.all():
            break
    return ["".join(o) for o in out]

def evaluate(generated, train_set):
    parsed = [(s, Chem.MolFromSmiles(s)) for s in generated]
    valid = [(s, m) for s, m in parsed if m is not None and m.GetNumHeavyAtoms() > 3]
    canon = {Chem.MolToSmiles(m) for _, m in valid}
    return {"validity": len(valid) / len(generated), "uniqueness": len(canon) / max(len(valid), 1),
            "novelty": len(canon - set(train_set)) / max(len(canon), 1)}, valid

rows = []
samples = {}
for T in [0.6, 0.8, 1.0, 1.2]:
    gen = sample(400, temperature=T)
    stats, valid = evaluate(gen, set(df.smiles))
    samples[T] = valid
    rows.append({"temperature": T, **stats})
pd.DataFrame(rows).set_index("temperature").round(3)
```

Low temperature gives valid but repetitive molecules; high temperature gives novelty at the cost of validity. This **validity–novelty trade-off** is the central metric of molecular generation.

```{code-cell} ipython3
valid = samples[0.8]
Draw.MolsToGridImage([m for _, m in valid[:16]], legends=[s[:24] for s, _ in valid[:16]],
                     molsPerRow=4, subImgSize=(190, 150))
```

## Why SELFIES exist

Every SELFIES string decodes to a valid molecule, so a generator cannot produce chemical nonsense {cite:p}`krenn2020`. The difference shows up as soon as you mutate a string at random.

```{code-cell} ipython3
def random_mutations(smi, n=200):
    out = []
    for _ in range(n):
        s = list(smi)
        i = rng.integers(len(s))
        s[i] = rng.choice(list("CNOScnos()=1234"))
        out.append("".join(s))
    return out

base = "CC(=O)Oc1ccccc1C(=O)O"
muts = random_mutations(base)
valid_frac = np.mean([Chem.MolFromSmiles(s) is not None for s in muts])
print(f"Random single-character mutations of a SMILES string: {valid_frac:.0%} still valid")
print("With SELFIES, by construction: 100 % valid (install selfies and try selfies.decoder)")
```

## A property-conditioned generator

To ask for a molecule *with a given property*, feed the target property into the model as it generates. Here the model sees the desired logP at every step.

```{code-cell} ipython3
logp = np.array([Crippen.MolLogP(Chem.MolFromSmiles(s)) for s in smiles], dtype=np.float32)
logp_n = torch.tensor((logp - logp.mean()) / logp.std()).unsqueeze(1)

class CondRNN(nn.Module):
    def __init__(self, n_vocab, d=128, layers=2):
        super().__init__()
        self.emb = nn.Embedding(n_vocab, d, padding_idx=0)
        self.cond = nn.Linear(1, d)
        self.rnn = nn.GRU(d, d, layers, batch_first=True, dropout=0.1)
        self.out = nn.Linear(d, n_vocab)
    def forward(self, x, c, h=None):
        z = self.emb(x) + self.cond(c).unsqueeze(1)
        z, h = self.rnn(z, h)
        return self.out(z), h

cmodel = CondRNN(len(chars) + 2)
opt = torch.optim.AdamW(cmodel.parameters(), lr=2e-3)
for epoch in range(30):
    perm = torch.randperm(len(data))
    for i in range(0, len(data), 128):
        idx = perm[i:i + 128]
        b, c = data[idx], logp_n[idx]
        logits, _ = cmodel(b[:, :-1], c)
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), b[:, 1:].reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()

@torch.no_grad()
def sample_cond(target_logp, n=300, temperature=0.9):
    cmodel.eval()
    c = torch.full((n, 1), (target_logp - logp.mean()) / logp.std())
    x = torch.ones(n, 1, dtype=torch.long); h = None
    out = [[] for _ in range(n)]; done = torch.zeros(n, dtype=torch.bool)
    for _ in range(L):
        logits, h = cmodel(x, c, h)
        probs = torch.softmax(logits[:, -1] / temperature, -1); probs[:, 0] = 0
        nxt = torch.multinomial(probs, 1)
        for k in range(n):
            if not done[k]:
                if nxt[k, 0].item() == 1: done[k] = True
                else: out[k].append(itos[nxt[k, 0].item()])
        x = nxt
        if done.all(): break
    mols = [Chem.MolFromSmiles("".join(o)) for o in out]
    return [m for m in mols if m is not None and m.GetNumHeavyAtoms() > 3]

fig, ax = plt.subplots(figsize=(7.5, 4.5))
for target, c in zip([-1.0, 1.0, 3.0, 5.0], ["#3b82f6", "#0f9d8a", "#f2a93b", "#e5487a"]):
    got = [Crippen.MolLogP(m) for m in sample_cond(target)]
    ax.hist(got, bins=25, alpha=0.55, color=c, label=f"asked for logP {target:+.0f} → got {np.mean(got):+.2f}")
    ax.axvline(target, color=c, ls="--")
ax.set(xlabel="logP of generated molecules", ylabel="count", title="Conditioning steers the generator")
ax.legend(fontsize=8); plt.show()
```

## Optimisation in the generator: a genetic algorithm

An alternative to learning a conditional model is to **search**: mutate and recombine molecules, keeping the best according to a scoring function. This is still one of the strongest baselines in molecular optimisation.

```{code-cell} ipython3
from rdkit.Chem import AllChem

MUTATIONS = [("[cH:1]", "[c:1]F"), ("[cH:1]", "[c:1]Cl"), ("[cH:1]", "[c:1]C"), ("[cH:1]", "[c:1]O"),
             ("[cH:1]", "[c:1]C#N"), ("[cH:1]", "[c:1]N"), ("[CX4H3:1]", "[C:1]C"),
             ("[c:1]F", "[cH:1]"), ("[c:1]C", "[cH:1]")]
RXNS = [AllChem.ReactionFromSmarts(f"{a}>>{b}") for a, b in MUTATIONS]

def mutate(m):
    rxn = RXNS[rng.integers(len(RXNS))]
    prods = rxn.RunReactants((m,))
    if not prods:
        return None
    p = prods[rng.integers(len(prods))][0]
    try:
        Chem.SanitizeMol(p)
        return p if p.GetNumHeavyAtoms() <= 30 else None
    except Exception:
        return None

def score(m):
    """Target: drug-like, logP near 2.5, with an aromatic nitrogen (our 'hinge binder')."""
    lp = Crippen.MolLogP(m)
    hinge = m.HasSubstructMatch(Chem.MolFromSmarts("[nX2]"))
    return QED.qed(m) + np.exp(-((lp - 2.5) / 1.2) ** 2) + 0.6 * hinge

population = [Chem.MolFromSmiles(s) for s in rng.choice(df.smiles, 60, replace=False)]
history = []
for generation in range(25):
    children = [c for c in (mutate(m) for m in population for _ in range(2)) if c is not None]
    pool = population + children
    seen, uniq = set(), []
    for m in pool:
        cs = Chem.MolToSmiles(m)
        if cs not in seen:
            seen.add(cs); uniq.append(m)
    scores = np.array([score(m) for m in uniq])
    keep = np.argsort(scores)[::-1][:60]
    population = [uniq[i] for i in keep]
    history.append(scores[keep].mean())

plt.plot(history, color="#0f9d8a", lw=2.5); plt.xlabel("generation"); plt.ylabel("mean score of the population")
plt.title("A genetic algorithm optimising a multi-objective score"); plt.show()
best = sorted(population, key=score, reverse=True)[:8]
Draw.MolsToGridImage(best, legends=[f"score {score(m):.2f} | logP {Crippen.MolLogP(m):.1f} | QED {QED.qed(m):.2f}" for m in best],
                     molsPerRow=4, subImgSize=(200, 160))
```

## Evaluating generated molecules

A generative model is only as good as its evaluation. Report, at minimum:

```{code-cell} ipython3
gen_set = samples[0.8]
gen_mols = [m for _, m in gen_set]
fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
train_fps = [fpgen.GetFingerprint(Chem.MolFromSmiles(s)) for s in rng.choice(df.smiles, 400, replace=False)]
gen_fps = [fpgen.GetFingerprint(m) for m in gen_mols]

metrics = {
    "validity": len(gen_mols) / 400,
    "uniqueness": len({Chem.MolToSmiles(m) for m in gen_mols}) / len(gen_mols),
    "novelty": len({Chem.MolToSmiles(m) for m in gen_mols} - set(df.smiles)) / len({Chem.MolToSmiles(m) for m in gen_mols}),
    "internal diversity": 1 - np.mean([np.mean(DataStructs.BulkTanimotoSimilarity(f, gen_fps)) for f in gen_fps]),
    "nearest-neighbour similarity to training": np.mean([max(DataStructs.BulkTanimotoSimilarity(f, train_fps)) for f in gen_fps]),
    "mean QED": float(np.mean([QED.qed(m) for m in gen_mols])),
}
pd.Series(metrics).round(3)
```

```{code-cell} ipython3
fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
for ax, (fn, label) in zip(axes, [(Descriptors.MolWt, "molecular weight"), (Crippen.MolLogP, "logP"), (QED.qed, "QED")]):
    ax.hist([fn(Chem.MolFromSmiles(s)) for s in df.smiles], bins=30, density=True, alpha=0.6, color="#6c4ad6", label="training")
    ax.hist([fn(m) for m in gen_mols], bins=30, density=True, alpha=0.6, color="#f2a93b", label="generated")
    ax.set_xlabel(label)
axes[0].set_ylabel("density"); axes[0].legend(); fig.suptitle("Does the generator match the training distribution?")
plt.tight_layout(); plt.show()
```

```{warning}
High scores on a scoring function are not a discovery. Generative models are notorious for exploiting their objectives, producing molecules that score beautifully and cannot be made, are unstable, or contain reactive groups. Always filter with synthetic accessibility, structural alerts and a chemist's eye, and remember that the same tools could propose harmful molecules, which is why the field discusses dual-use risk openly {cite:p}`urbina2022`.
```

## Generating crystals

For inorganic materials the same logic applies to composition and lattice. A simple generator proposes charge-balanced compositions and checks them against a stability model, which is the screening loop of Chapter 41 run in reverse.

```{code-cell} ipython3
from pymatgen.core import Composition, Element
CATIONS = {"Li": 1, "Na": 1, "K": 1, "Mg": 2, "Ca": 2, "Zn": 2, "Al": 3, "Ga": 3, "Ti": 4}
ANIONS = {"O": -2, "S": -2, "Se": -2, "N": -3, "F": -1, "Cl": -1}
proposals = []
for _ in range(3000):
    (a, ca), (b, cb) = [list(CATIONS.items())[rng.integers(len(CATIONS))], list(ANIONS.items())[rng.integers(len(ANIONS))]]
    for na in range(1, 5):
        nb = -na * ca / cb
        if nb.is_integer() and 1 <= nb <= 4:
            f = f"{a}{na if na > 1 else ''}{b}{int(nb) if nb > 1 else ''}"
            proposals.append({"formula": f, "dchi": abs(Element(b).X - Element(a).X),
                              "mean_r": float((na * Element(a).atomic_radius + nb * Element(b).atomic_radius) / (na + nb))})
prop = pd.DataFrame(proposals).drop_duplicates("formula")
prop["predicted_gap"] = np.clip(2.6 * prop.dchi - 1.1 * prop.mean_r + 0.8, 0, None)
hits = prop.query("1.1 <= predicted_gap <= 1.7")
print(f"{len(prop)} unique charge-balanced compositions generated, {len(hits)} with a useful predicted gap")
hits.sample(min(10, len(hits)), random_state=0).round(2)
```

## Exercises

**★ 1.** Sample 1 000 molecules at temperature 0.7 and 1.3 and compare validity and internal diversity.

**★★ 2.** Condition the generator on TPSA instead of logP. Which property is easier to control, and why?

**★★★ 3.** Add a synthetic-accessibility penalty to the genetic algorithm's score and see how the winning molecules change.
