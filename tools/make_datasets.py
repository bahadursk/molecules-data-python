"""Generate the teaching datasets in book/data/.

    python tools/make_datasets.py

Every dataset in this book is either generated here or inside a notebook, so
the book carries no third-party data and no licensing restrictions. The
molecules are real chemical structures built by combining common scaffolds and
substituents; the *property values* are simulated from transparent,
physically motivated formulas plus noise, and are labelled "_sim". They behave
like real data for teaching, but must never be used as experimental values.
"""
from pathlib import Path
import itertools

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")
OUT = Path(__file__).resolve().parents[1] / "book" / "data"
OUT.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(2026)

SCAFFOLDS = [
    "c1ccc({R1})cc1{R2}", "c1cc({R1})ccc1{R2}", "c1ccc2cc({R1})ccc2c1{R2}", "c1cc({R1})ncc1{R2}",
    "c1cc({R1})sc1{R2}", "c1cc({R1})oc1{R2}", "c1ccc2[nH]c({R1})cc2c1{R2}", "c1ccc2oc({R1})nc2c1{R2}",
    "c1nc({R1})cc({R2})n1", "C1CCN(CC1){R1}", "C1CN(CCN1{R2}){R1}", "C1CCC(CC1){R1}",
    "O=C(N{R1})c1ccccc1{R2}", "O=C(O{R1})c1ccc({R2})cc1", "c1ccc(cc1)-c1ccc({R1})cc1{R2}",
    "c1ccc(cc1)Oc1ccc({R1})cc1", "O=C1NC(=O)C({R1})N1", "c1cn({R1})cn1", "c1ccc2ncccc2c1{R1}",
    "O=S(=O)(N{R1})c1ccc({R2})cc1", "C(=O)(N{R1}){R2}", "c1csc(n1){R1}", "c1ccc2c(c1)CCN2{R1}",
]
SUBSTITUENTS = ["", "C", "CC", "C(C)C", "C(C)(C)C", "CCCC", "O", "OC", "OCC", "N", "NC", "N(C)C", "F", "Cl", "Br",
                "C(F)(F)F", "C#N", "[N+](=O)[O-]", "C(=O)O", "C(=O)OC", "C(=O)N", "C(=O)C", "S(=O)(=O)C", "c1ccccc1",
                "C1CC1", "OC(F)(F)F", "CO", "CCN", "NC(=O)C", "c1ccncc1"]


def build_library():
    seen, rows = set(), []
    for scaf in SCAFFOLDS:
        for r1, r2 in itertools.product(SUBSTITUENTS, SUBSTITUENTS):
            if "{R2}" not in scaf and r2:
                continue
            smi = scaf.replace("{R1}", f"({r1})" if r1 else "").replace("{R2}", f"({r2})" if r2 else "")
            smi = smi.replace("()", "")
            m = Chem.MolFromSmiles(smi)
            if m is None or m.GetNumHeavyAtoms() < 5:
                continue
            can = Chem.MolToSmiles(m)
            if can not in seen:
                seen.add(can)
                rows.append(can)
    return rows


ALERTS = {"nitro": "[N+](=O)[O-]", "aryl bromide": "cBr", "aniline": "c[NH2]", "thiophene": "c1ccsc1",
          "Michael acceptor": "C=CC=O", "aromatic amine N-Me": "cN(C)C"}


def molecules():
    smiles = build_library()
    rows = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        logp, mw = Crippen.MolLogP(m), Descriptors.MolWt(m)
        rb = rdMolDescriptors.CalcNumRotatableBonds(m)
        arom = sum(a.GetIsAromatic() for a in m.GetAtoms()) / m.GetNumHeavyAtoms()
        tpsa = rdMolDescriptors.CalcTPSA(m)
        hbd = rdMolDescriptors.CalcNumHBD(m)
        # Solubility: a linear ESOL-style relation (Delaney 2004) plus a curved H-bonding term and noise
        logS = 0.16 - 0.63 * logp - 0.0062 * mw + 0.066 * rb - 0.74 * arom + 0.12 * hbd - 0.002 * (tpsa - 60) ** 2 / 10
        logS += rng.normal(0, 0.35)
        # A fictitious kinase: rewards an H-bond donor next to an aromatic N, a lipophilic group, moderate size
        has_hinge = m.HasSubstructMatch(Chem.MolFromSmarts("[nX2]:c[NX3;H1,H2]")) or m.HasSubstructMatch(Chem.MolFromSmarts("[nH]"))
        has_halo = m.HasSubstructMatch(Chem.MolFromSmarts("c[F,Cl,Br,$(C(F)(F)F)]"))
        pic50 = 5.3 + 1.4 * has_hinge + 0.6 * has_halo + 0.35 * np.clip(logp, 0, 4) - 0.004 * abs(mw - 320) + rng.normal(0, 0.4)
        alerts = [k for k, v in ALERTS.items() if m.HasSubstructMatch(Chem.MolFromSmarts(v))]
        p_tox = 1 / (1 + np.exp(-(-2.2 + 1.8 * len(alerts) + 0.35 * (logp - 2))))
        rows.append({"smiles": s, "scaffold": MurckoScaffold.MurckoScaffoldSmiles(mol=m), "MW": round(mw, 2),
                     "logP": round(logp, 3), "TPSA": round(tpsa, 2), "HBD": hbd, "RotB": rb,
                     "logS_sim": round(logS, 3), "pIC50_sim": round(pic50, 3),
                     "toxic_sim": int(rng.random() < p_tox), "alerts": ";".join(alerts)})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "molecules_sim.csv", index=False)
    return df


ARYL_HALIDES = {"PhCl": "Clc1ccccc1", "PhBr": "Brc1ccccc1", "PhI": "Ic1ccccc1", "4-MeO-PhBr": "COc1ccc(Br)cc1",
                "4-CF3-PhCl": "FC(F)(F)c1ccc(Cl)cc1", "3-pyridyl-Br": "Brc1cccnc1", "2-thienyl-Br": "Brc1cccs1",
                "4-Et-PhI": "CCc1ccc(I)cc1"}
LIGANDS = {"L1": (0.2, 160), "L2": (0.5, 175), "L3": (0.8, 190), "L4": (0.65, 205)}      # (electron richness, cone angle)
BASES = {"K2CO3": 10.3, "Cs2CO3": 10.4, "KOtBu": 17.0, "DBU": 13.5, "K3PO4": 12.3}
ADDITIVES = ["none", "isoxazole A", "isoxazole B", "ester C", "pyridine D", "water"]


def reactions():
    rows = []
    halide_reactivity = {"Cl": 0.0, "Br": 0.6, "I": 0.9}
    for (hn, hs), (ln, (rich, cone)), (bn, pka), add in itertools.product(ARYL_HALIDES.items(), LIGANDS.items(), BASES.items(), ADDITIVES):
        X = "I" if "I" in hs else "Br" if "Br" in hs else "Cl"
        ewg = 0.5 if "C(F)(F)F" in hs or "n" in hs else (-0.3 if "CO" in hs else 0)
        z = (-1.0 + 2.2 * halide_reactivity[X] + 1.4 * rich * (1.2 if X == "Cl" else 1) - ((cone - 185) / 25) ** 2
             + 0.35 * ewg + 0.12 * (pka - 12) - (0.9 if add == "pyridine D" else 0) - (0.4 if add == "isoxazole B" else 0)
             + (0.2 if add == "water" and bn == "K3PO4" else 0))
        y = 100 / (1 + np.exp(-z)) + rng.normal(0, 4)
        rows.append({"aryl_halide": hn, "aryl_halide_smiles": hs, "ligand": ln, "ligand_electron_richness": rich,
                     "ligand_cone_angle": cone, "base": bn, "base_pKa": pka, "additive": add,
                     "yield_sim": round(float(np.clip(y, 0, 100)), 1)})
    pd.DataFrame(rows).to_csv(OUT / "cn_coupling_hte_sim.csv", index=False)


def materials():
    from pymatgen.core import Element
    A = ["Li", "Na", "K", "Rb", "Cs", "Mg", "Ca", "Sr", "Ba", "Zn", "Cd", "Al", "Ga", "In"]
    B = ["O", "S", "Se", "Te", "F", "Cl", "Br", "I", "N", "P", "As"]
    rows = []
    for a, b in itertools.product(A, B):
        ea, eb = Element(a), Element(b)
        for na, nb in [(1, 1), (1, 2), (2, 1), (2, 3), (3, 2)]:
            charge_a = ea.common_oxidation_states[0] if ea.common_oxidation_states else 1
            charge_b = min(eb.common_oxidation_states) if eb.common_oxidation_states else -1
            balanced = na * charge_a + nb * charge_b == 0
            dchi = abs(eb.X - ea.X)
            mean_r = (na * ea.atomic_radius + nb * eb.atomic_radius) / (na + nb)
            gap = max(0.0, 2.6 * dchi - 1.1 * float(mean_r) + 0.8 * balanced + rng.normal(0, 0.3)) if balanced else max(0.0, rng.normal(0.1, 0.2))
            ef = -0.9 * dchi ** 1.3 * balanced + 0.25 * (not balanced) + rng.normal(0, 0.08)
            rows.append({"formula": f"{a}{na if na > 1 else ''}{b}{nb if nb > 1 else ''}", "A": a, "B": b, "nA": na, "nB": nb,
                         "chi_A": ea.X, "chi_B": eb.X, "r_A": float(ea.atomic_radius), "r_B": float(eb.atomic_radius),
                         "Z_A": ea.Z, "Z_B": eb.Z, "charge_balanced": int(balanced),
                         "band_gap_sim_eV": round(gap, 3), "formation_energy_sim_eV_atom": round(ef, 3)})
    pd.DataFrame(rows).to_csv(OUT / "binary_materials_sim.csv", index=False)


POLYMER_UNITS = {
    "polyethylene": "*CC*", "polypropylene": "*CC(C)*", "polystyrene": "*CC(c1ccccc1)*", "PVC": "*CC(Cl)*",
    "PMMA": "*CC(C)(C(=O)OC)*", "PVA": "*CC(O)*", "PAN": "*CC(C#N)*", "PTFE": "*C(F)(F)C(F)(F)*",
    "PEO": "*CCO*", "PET-like": "*OCCOC(=O)c1ccc(cc1)C(=O)*", "nylon-6-like": "*NCCCCCC(=O)*",
    "PVDF": "*CC(F)(F)*", "poly(4-methylstyrene)": "*CC(c1ccc(C)cc1)*", "poly(vinyl acetate)": "*CC(OC(C)=O)*",
    "poly(acrylic acid)": "*CC(C(=O)O)*", "polyisoprene": "*CC(C)=CC*", "polybutadiene": "*CC=CC*",
    "poly(4-vinylpyridine)": "*CC(c1ccncc1)*", "poly(tBu acrylate)": "*CC(C(=O)OC(C)(C)C)*", "PC-like": "*Oc1ccc(cc1)C(C)(C)c1ccc(cc1)OC(=O)*",
}


# Approximate glass-transition temperatures (°C), rounded textbook values for well-known polymers
TG_APPROX = {"polyethylene": -120, "polypropylene": -10, "polystyrene": 100, "PVC": 80, "PMMA": 105, "PVA": 85,
             "PAN": 95, "PTFE": 115, "PEO": -60, "PET-like": 70, "nylon-6-like": 50, "PVDF": -35,
             "poly(4-methylstyrene)": 105, "poly(vinyl acetate)": 30, "poly(acrylic acid)": 105, "polyisoprene": -70,
             "polybutadiene": -90, "poly(4-vinylpyridine)": 140, "poly(tBu acrylate)": 40, "PC-like": 150}


def _poly_features(smi):
    m = Chem.MolFromSmiles(smi)
    return [rdMolDescriptors.CalcNumAromaticRings(m), rdMolDescriptors.CalcNumRotatableBonds(m),
            rdMolDescriptors.CalcTPSA(m), int(m.HasSubstructMatch(Chem.MolFromSmarts("[CX4;H0](C)(C)"))),
            int(m.HasSubstructMatch(Chem.MolFromSmarts("C=C"))), m.GetNumAtoms() - 2]


def polymers():
    """Named polymers carry rounded textbook Tg values; the extra vinyl polymers get simulated values
    from a linear model fitted to those, plus noise."""
    known = [(n, POLYMER_UNITS[n], TG_APPROX[n]) for n in POLYMER_UNITS]
    X = np.array([_poly_features(s) for _, s, _ in known] , dtype=float)
    y = np.array([t for *_, t in known], dtype=float)
    Xa = np.column_stack([X, np.ones(len(X))])
    coef, *_ = np.linalg.lstsq(Xa, y, rcond=None)
    out = [{"name": n, "repeat_unit": s, "Tg_C": t, "source": "approximate textbook value"} for n, s, t in known]
    side_chains = ["C", "CC", "CCCC", "CCCCCC", "c1ccccc1", "c1ccc(Cl)cc1", "Cl", "F", "O", "C(=O)OC", "C(=O)OCC",
                   "C(=O)OCCCC", "C#N", "OC(=O)C", "C(=O)N", "c1ccncc1", "C(=O)OC(C)(C)C", "OC"]
    for sc in side_chains:
        for tag, backbone in [("", "*CC({})*"), ("α-methyl ", "*CC(C)({})*")]:
            smi = backbone.format(sc)
            if smi in POLYMER_UNITS.values():
                continue
            tg = float(np.dot(np.append(_poly_features(smi), 1.0), coef) + rng.normal(0, 15))
            out.append({"name": f"{tag}poly(vinyl-{sc})", "repeat_unit": smi, "Tg_C": round(tg, 1), "source": "simulated"})
    pd.DataFrame(out).drop_duplicates("repeat_unit").to_csv(OUT / "polymers_sim.csv", index=False)


if __name__ == "__main__":
    df = molecules()
    reactions()
    materials()
    polymers()
    for f in sorted(OUT.glob("*.csv")):
        print(f.name, sum(1 for _ in open(f)) - 1, "rows")
