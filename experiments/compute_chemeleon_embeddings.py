"""Compute CheMeleon embeddings for every molecule in the benchmark oracles (run in the catbots-gnn env).

Writes src/data/embeddings/chemeleon_mp.npz with arrays ``smiles``, ``inchikey14`` and ``X`` (float32,
n_molecules x 2048), so the core pipeline can use the embeddings without PyTorch.

    conda activate catbots-gnn
    python experiments/compute_chemeleon_embeddings.py
"""

from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from chemprop import data, featurizers, nn

from catbots import NONE, list_benchmarks, load_benchmark
from catbots.chem import inchikey14
from catbots.paths import EMBEDDING_DIR

WEIGHTS_URL = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"  # cite arXiv:2506.15792
MOLECULE_FACTORS = {"ligand", "electrophile", "nucleophile", "base", "solvent", "metal_complex"}


def load_chemeleon(device):
    ckpt = Path.home() / ".chemprop" / "chemeleon_mp.pt"
    if not ckpt.exists():
        ckpt.parent.mkdir(exist_ok=True)
        urlretrieve(WEIGHTS_URL, ckpt)
    state = torch.load(ckpt, weights_only=True)
    mp = nn.BondMessagePassing(**state["hyper_parameters"])
    mp.load_state_dict(state["state_dict"])
    return mp.to(device).eval(), nn.MeanAggregation()


def main():
    smiles = set()
    for name in list_benchmarks():
        o = load_benchmark(name)
        for f in MOLECULE_FACTORS & set(o.factors):
            smiles.update(s for s in o.levels(f) if s != NONE)
    smiles = sorted(smiles)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mp, agg = load_chemeleon(device)
    featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer(atom_featurizer=featurizers.MultiHotAtomFeaturizer.v2())

    ok, failed = [], []
    for s in smiles:  # featurise one by one so a single unparsable complex does not drop the batch
        try:
            data.MoleculeDatapoint.from_smi(s)
            featurizer(data.MoleculeDatapoint.from_smi(s).mol)
            ok.append(s)
        except Exception as exc:  # noqa: BLE001 - report and skip
            failed.append((s, type(exc).__name__))
    dset = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in ok], featurizer=featurizer)
    out = []
    with torch.no_grad():
        for batch in data.build_dataloader(dset, shuffle=False, batch_size=128):
            bmg = batch[0]
            bmg.to(device)
            out.append(agg(mp(bmg), bmg.batch).cpu().numpy())
    X = np.vstack(out).astype(np.float32)

    EMBEDDING_DIR.mkdir(parents=True, exist_ok=True)
    path = EMBEDDING_DIR / "chemeleon_mp.npz"
    np.savez_compressed(path, smiles=np.array(ok), inchikey14=np.array([inchikey14(s) or "" for s in ok]), X=X)
    print(f"{len(ok)} molecules embedded ({X.shape[1]}-d) on {device} -> {path}")
    if failed:
        print(f"{len(failed)} molecules could not be featurised:")
        for s, err in failed:
            print(f"  {err}: {s}")


if __name__ == "__main__":
    main()
