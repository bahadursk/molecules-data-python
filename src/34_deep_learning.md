---
jupytext:
  text_representation:
    format_name: myst
kernelspec:
  name: python3
  display_name: Python 3
---

# Deep Learning with PyTorch

```{admonition} 🔴 Advanced · graduate & research
:class: level-3
Assumes Chapters 12, 31 and 32, and a little calculus.
```

A **neural network** is a model built from many simple units arranged in layers. Each unit computes a weighted sum of its inputs and passes it through a simple non-linear function; stacking layers lets the network represent very complicated relationships. A network with several hidden layers is called **deep**, and its great strength is that it can learn its own features from raw inputs such as graphs, sequences and images, rather than relying on hand-made descriptors (Chapter 32). Training a network means adjusting its weights, often millions of them, to reduce the prediction error on the training data, which is done by **gradient descent**: compute how the error changes with each weight and nudge every weight a little downhill. This chapter builds networks from the ground up in **PyTorch**: tensors and automatic differentiation, a feed-forward network on fingerprints, a graph neural network written from scratch, a small transformer for SMILES, and uncertainty with ensembles.

```{code-cell} ipython3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

torch.manual_seed(0)
DATA_URL = "https://raw.githubusercontent.com/bahadursk/molecules-data-python/main/book/data/"
def load_data(name):
    local = Path("../data") / name
    return pd.read_csv(local if local.exists() else DATA_URL + name)
df = load_data("molecules_sim.csv")
print("PyTorch", torch.__version__)
```

## Tensors and autograd

PyTorch's basic object is the **tensor**, a multidimensional array much like a NumPy array (Chapter 12) but able to run on a GPU and to track the operations performed on it. If a tensor is created with `requires_grad=True`, PyTorch records every calculation that uses it, and calling `.backward()` on a final result computes the derivative of that result with respect to the tensor by the chain rule. This is **automatic differentiation** (autograd), and it means you never have to work out a gradient by hand. A one-line function shows it: the derivative of $x^2$ at $x = 3$ is $2x = 6$.

```{code-cell} ipython3
x_g = torch.tensor(3.0, requires_grad=True)
f_g = x_g ** 2
f_g.backward()
print(x_g.grad)
```

Gradient descent uses these derivatives to fit parameters. Starting from a poor guess for a decay constant $k$, each step computes the **loss** (the mean squared error between the model and the data), its gradient with respect to $k$, and moves $k$ a small amount (the **learning rate**) against the gradient. An optimiser object such as `SGD` performs the update, and `zero_grad()` clears the previous gradient before each new one is computed.

```{code-cell} ipython3
# Fit k in the Arrhenius-like model y = exp(-k x) by gradient descent, with gradients computed automatically
x = torch.linspace(0, 5, 50)
y = torch.exp(-0.7 * x) + 0.02 * torch.randn(50)
k = torch.tensor(0.1, requires_grad=True)
opt = torch.optim.SGD([k], lr=0.5)
history = []
for step in range(60):
    loss = ((torch.exp(-k * x) - y) ** 2).mean()
    opt.zero_grad(); loss.backward(); opt.step()
    history.append((k.item(), loss.item()))
print(f"fitted k = {k.item():.3f} (true 0.7)")
plt.plot([h[0] for h in history], color="#6c4ad6"); plt.xlabel("step"); plt.ylabel("k"); plt.title("Gradient descent at work"); plt.show()
```

`loss.backward()` computes ∂loss/∂k by the chain rule, through every operation, automatically. That single idea trains every network below.

## A feed-forward network on fingerprints

The simplest network is a **multilayer perceptron** (MLP), or feed-forward network. A **linear layer** (`nn.Linear`) multiplies its input vector by a weight matrix and adds a bias; an **activation function** such as ReLU, $\max(0, x)$, follows it, because without a non-linearity a stack of linear layers would collapse into a single linear model. **Dropout** randomly switches off a fraction of units during training, which discourages the network from relying on any single feature and reduces overfitting. In PyTorch a model is a class derived from `nn.Module` (Chapter 10) whose `forward` method says how the input flows through the layers.

A single layer applied to one fingerprint-like input shows the shapes involved:

```{code-cell} ipython3
layer = nn.Linear(4, 2)                    # 4 inputs -> 2 outputs
x_in = torch.tensor([[1.0, 0.0, 1.0, 1.0]])
print(layer(x_in).shape, torch.relu(layer(x_in)))
```

Training loops over **epochs** (complete passes through the training data), and within each epoch over **mini-batches** of 64 molecules, a compromise between the noise of single examples and the cost of the whole dataset. The data are split three ways: the network is trained on the training set, its progress is watched on a **validation** set to choose settings and detect overfitting, and the test set is used once at the end. Count fingerprints are compressed with `log1p` so that large counts do not dominate. `AdamW` is an adaptive optimiser that works well with default settings, and its `weight_decay` gently shrinks the weights.

```{code-cell} ipython3
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
X = np.array([gen.GetCountFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in df.smiles], dtype=np.float32)
X = np.log1p(X)
y = df.pIC50_sim.values.astype(np.float32)
rng = np.random.default_rng(0)
idx = rng.permutation(len(X)); n_tr, n_va = int(0.7 * len(X)), int(0.85 * len(X))
tr, va, te = idx[:n_tr], idx[n_tr:n_va], idx[n_va:]
Xt, yt = torch.tensor(X), torch.tensor(y)

class MLP(nn.Module):
    def __init__(self, n_in, hidden=256, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, hidden), nn.ReLU(), nn.Dropout(dropout),
                                 nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def train(model, Xtr, ytr, Xva, yva, epochs=80, lr=1e-3, wd=1e-4, batch=64):
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    log = []
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr))
        for i in range(0, len(Xtr), batch):
            b = perm[i:i + batch]
            loss = nn.functional.mse_loss(model(Xtr[b]), ytr[b])
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            log.append((nn.functional.mse_loss(model(Xtr), ytr).item(), nn.functional.mse_loss(model(Xva), yva).item()))
    return np.array(log)

mlp = MLP(X.shape[1])
log = train(mlp, Xt[tr], yt[tr], Xt[va], yt[va])
plt.plot(log[:, 0], color="#0f9d8a", label="train"); plt.plot(log[:, 1], color="#e5487a", label="validation")
plt.xlabel("epoch"); plt.ylabel("MSE"); plt.legend(); plt.title("Learning curves"); plt.show()
with torch.no_grad():
    pred = mlp(Xt[te]).numpy()
from sklearn.metrics import r2_score
print(f"MLP test R² = {r2_score(y[te], pred):.3f}")
```

The learning curves show the training error falling steadily while the validation error levels off; a widening gap between them is the signature of overfitting.

## A graph neural network from scratch

Fingerprints are fixed by hand before training. A **graph neural network** (GNN) instead learns directly from the molecular graph of Chapter 32: one feature vector per atom and an adjacency matrix. In a **message-passing** network {cite:p}`gilmer2017` each atom repeatedly gathers information from its bonded neighbours:

$$\mathbf h_v^{(t+1)} = \sigma\Big(W_1 \mathbf h_v^{(t)} + W_2 \sum_{u \in \mathcal N(v)} \mathbf h_u^{(t)}\Big)$$

After a few rounds, the atom vectors are summed into one molecule vector and passed to a regressor. Each round lets information travel one bond further, so after three rounds every atom "knows" about its neighbourhood up to three bonds away, much like a Morgan fingerprint of radius 3, except that what to record is learned. Multiplying the atom-feature matrix by the adjacency matrix, `A @ h`, adds up each atom's neighbours' vectors in one operation:

```{code-cell} ipython3
A_demo = torch.tensor([[0., 1., 0.], [1., 0., 1.], [0., 1., 0.]])   # a three-atom chain
h_demo = torch.tensor([[1.], [10.], [100.]])                           # one number per atom
print((A_demo @ h_demo).squeeze().tolist())                            # sum of each atom's neighbours
```

To train on many molecules at once, their graphs are stacked into one large **block-diagonal** adjacency matrix, so that atoms of different molecules never exchange messages, and an `owner` index records which molecule each atom belongs to. Summing atom vectors per molecule (`index_add_`) is the **readout** or pooling step.

```{code-cell} ipython3
ATOMS = ["C", "N", "O", "S", "F", "Cl", "Br"]
def featurise(smi):
    m = Chem.MolFromSmiles(smi)
    h = [[a.GetSymbol() == s for s in ATOMS] + [a.GetSymbol() not in ATOMS, a.GetIsAromatic(), a.GetTotalNumHs() / 3,
          a.GetDegree() / 4, a.IsInRing(), a.GetFormalCharge()] for a in m.GetAtoms()]
    A = Chem.GetAdjacencyMatrix(m)
    return torch.tensor(h, dtype=torch.float32), torch.tensor(A, dtype=torch.float32)

graphs = [featurise(s) for s in df.smiles]

def batch_graphs(ids):
    """Stack several molecules into one block-diagonal graph."""
    hs, As, owner = [], [], []
    for k, i in enumerate(ids):
        h, A = graphs[i]; hs.append(h); As.append(A); owner += [k] * len(h)
    return torch.cat(hs), torch.block_diag(*As), torch.tensor(owner)

class MPNN(nn.Module):
    def __init__(self, n_feat, hidden=96, steps=3):
        super().__init__()
        self.embed = nn.Linear(n_feat, hidden)
        self.self_w = nn.ModuleList([nn.Linear(hidden, hidden) for _ in range(steps)])
        self.nbr_w = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(steps)])
        self.readout = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(), nn.Linear(64, 1))
    def forward(self, h, A, owner, n_mol):
        h = torch.relu(self.embed(h))
        for Ws, Wn in zip(self.self_w, self.nbr_w):
            h = torch.relu(Ws(h) + Wn(A @ h))
        pooled = torch.zeros(n_mol, h.shape[1]).index_add_(0, owner, h)      # sum over atoms of each molecule
        return self.readout(pooled).squeeze(-1)

gnn = MPNN(graphs[0][0].shape[1])
opt = torch.optim.AdamW(gnn.parameters(), lr=2e-3, weight_decay=1e-4)
val_curve = []
for epoch in range(40):
    gnn.train()
    for b in np.array_split(rng.permutation(tr), len(tr) // 64):
        h, A, owner = batch_graphs(b)
        loss = nn.functional.mse_loss(gnn(h, A, owner, len(b)), yt[b])
        opt.zero_grad(); loss.backward(); opt.step()
    gnn.eval()
    with torch.no_grad():
        h, A, owner = batch_graphs(va)
        val_curve.append(nn.functional.mse_loss(gnn(h, A, owner, len(va)), yt[va]).item())
with torch.no_grad():
    h, A, owner = batch_graphs(te)
    pred_gnn = gnn(h, A, owner, len(te)).numpy()
print(f"GNN test R² = {r2_score(y[te], pred_gnn):.3f}")
plt.plot(val_curve, color="#6c4ad6"); plt.xlabel("epoch"); plt.ylabel("validation MSE"); plt.title("Training the message-passing network"); plt.show()
```

Production libraries (PyTorch Geometric, DGL, Chemprop) add edge features, attention and efficient sparse batching, but the core idea is exactly this loop. 3D-aware variants such as SchNet {cite:p}`schutt2018` use interatomic distances instead of bonds.

### What did the network learn?

Because the readout sums atom vectors, the final prediction is (before the last non-linear step) a sum of per-atom terms. Atom-level contributions can be read straight from the model by passing each atom's final vector through the readout.

```{code-cell} ipython3
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D
from IPython.display import Image

def atom_contributions(smi):
    h, A = featurise(smi)
    with torch.no_grad():
        z = torch.relu(gnn.embed(h))
        for Ws, Wn in zip(gnn.self_w, gnn.nbr_w):
            z = torch.relu(Ws(z) + Wn(A @ z))
        per_atom = gnn.readout[2](torch.relu(gnn.readout[0](z))).squeeze(-1)
    return per_atom.numpy()

smi = df.loc[te[np.argmax(pred_gnn)], "smiles"]
m = Chem.MolFromSmiles(smi)
w = atom_contributions(smi); w = w - w.mean()
d = rdMolDraw2D.MolDraw2DCairo(420, 320)
SimilarityMaps.GetSimilarityMapFromWeights(m, list(map(float, w)), colorMap="PiYG", draw2d=d)
d.FinishDrawing(); print(smi); Image(d.GetDrawingText())
```

## A tiny transformer for SMILES

Transformers treat SMILES as sentences, and attention lets every token look at every other token. The **transformer** is the architecture behind modern language models. Its key mechanism, **self-attention**, lets each token compute how relevant every other token in the sequence is to it and combine their information accordingly, so a ring-closure digit can "attend" to its partner far along the string. The pieces are: tokenisation (Chapter 32) and an **embedding** layer that turns each token id into a learned vector; a **positional** vector added to each, because attention on its own ignores order; several encoder layers; and a prediction head. Sequences are padded to the same length with token 0, and a padding mask tells attention to ignore those positions. A special first token (CLS) collects information from the whole sequence and is used for the prediction.

An embedding is simply a learnable lookup table:

```{code-cell} ipython3
emb_demo = nn.Embedding(5, 3)                 # 5 possible tokens, each mapped to a 3-number vector
print(emb_demo(torch.tensor([2, 2, 4])).shape)  # same token -> same vector
```

```{code-cell} ipython3
import re
TOK = re.compile(r"(\[[^\]]+]|Br|Cl|@@|[BCNOPSFIbcnops]|\(|\)|=|#|-|\+|/|\\|@|\d)")
vocab = {t: i + 2 for i, t in enumerate(sorted({t for s in df.smiles for t in TOK.findall(s)}))}   # 0 pad, 1 CLS
max_len = 1 + max(len(TOK.findall(s)) for s in df.smiles)
def encode(s):
    ids = [1] + [vocab[t] for t in TOK.findall(s)]
    return ids + [0] * (max_len - len(ids))
S = torch.tensor([encode(s) for s in df.smiles])

class SmilesTransformer(nn.Module):
    def __init__(self, n_vocab, d=64, heads=4, layers=2):
        super().__init__()
        self.tok = nn.Embedding(n_vocab, d, padding_idx=0)
        self.pos = nn.Parameter(torch.randn(max_len, d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, heads, dim_feedforward=128, batch_first=True, dropout=0.1)
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Linear(d, 1)
    def forward(self, s):
        z = self.enc(self.tok(s) + self.pos, src_key_padding_mask=(s == 0))
        return self.head(z[:, 0]).squeeze(-1)                     # use the CLS token

tfm = SmilesTransformer(len(vocab) + 2)
opt = torch.optim.AdamW(tfm.parameters(), lr=1e-3)
for epoch in range(25):
    tfm.train()
    for b in np.array_split(rng.permutation(tr), len(tr) // 64):
        loss = nn.functional.mse_loss(tfm(S[b]), yt[b]); opt.zero_grad(); loss.backward(); opt.step()
tfm.eval()
with torch.no_grad():
    print(f"Transformer test R² = {r2_score(y[te], tfm(S[te]).numpy()):.3f}")
```

## Uncertainty with deep ensembles

A single network gives a prediction but no indication of how much to trust it. A **deep ensemble** estimates that trust cheaply. Train several networks from different random starts. Where they disagree, be cautious. Networks trained from different random initial weights end up in different solutions that agree where the training data are dense and diverge where they are sparse, so the standard deviation across the ensemble acts as an uncertainty estimate. The check below sorts the test molecules by that spread into five groups; if the uncertainty is meaningful, the most confident group should have the smallest actual errors. Chapter 51 discusses uncertainty in depth.

```{code-cell} ipython3
ens_preds = []
for seed in range(5):
    torch.manual_seed(seed)
    net = MLP(X.shape[1]); train(net, Xt[tr], yt[tr], Xt[va], yt[va], epochs=40)
    net.eval()
    with torch.no_grad():
        ens_preds.append(net(Xt[te]).numpy())
ens = np.array(ens_preds); mu, sd = ens.mean(0), ens.std(0)
err = np.abs(mu - y[te])
order = np.argsort(sd)
bins = np.array_split(order, 5)
plt.bar(range(5), [err[b].mean() for b in bins], color=plt.cm.magma(np.linspace(0.2, 0.8, 5)))
plt.xticks(range(5), ["most\nconfident", "", "", "", "least\nconfident"])
plt.ylabel("mean absolute error"); plt.title(f"Ensemble spread predicts error (test R² = {r2_score(y[te], mu):.3f})")
plt.show()
```

## Comparing everything

The honest end of any modelling exercise is a comparison on the same test set, including a strong classical baseline.

```{code-cell} ipython3
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(300, n_jobs=-1, random_state=0).fit(X[tr], y[tr])
with torch.no_grad():
    res = {"random forest (fingerprints)": r2_score(y[te], rf.predict(X[te])), "MLP (fingerprints)": r2_score(y[te], pred),
           "GNN (graph)": r2_score(y[te], pred_gnn), "Transformer (SMILES)": r2_score(y[te], tfm(S[te]).numpy()),
           "MLP ensemble": r2_score(y[te], mu)}
pd.Series(res).sort_values().plot.barh(color="#6c4ad6", figsize=(7, 3)); plt.xlabel("test R²"); plt.show()
```

```{admonition} 🤖 ML corner
:class: ml-corner
With a few thousand molecules, a random forest on fingerprints is a very strong baseline that deep models often fail to beat. Deep learning shines with large datasets, raw inputs (spectra, images, 3D structures) and multitask problems. Always compare against a simple baseline.
```

```{admonition} Summary
:class: tip
- Tensors are GPU-capable arrays; autograd computes gradients with `.backward()`; optimisers update weights by gradient descent.
- An MLP stacks linear layers and non-linear activations; dropout and weight decay limit overfitting.
- Train in mini-batches over epochs, watch a validation set, test once.
- GNNs pass messages along bonds (`A @ h`) and pool atom vectors into a molecule vector.
- Transformers embed tokens, add positions and use self-attention.
- Deep ensembles give uncertainty estimates; always compare with a random-forest baseline.
```

## Exercises

**★ 1.** Change the MLP hidden size to 32 and 1 024. How does the validation curve change?

**★★ 2.** Add bond-type information to the GNN by using separate adjacency matrices for single, double and aromatic bonds.

**★★★ 3.** Train the transformer as a classifier for `toxic_sim` and visualise its attention weights for a nitro-containing molecule.
