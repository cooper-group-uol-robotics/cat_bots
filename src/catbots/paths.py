"""Locations of the data shipped with the repository.

Set the ``CATBOTS_DATA`` environment variable to point at a different data
directory (it must have the same layout as ``src/data``).
"""

import os
from pathlib import Path

DATA_DIR = Path(os.environ.get("CATBOTS_DATA", Path(__file__).resolve().parents[1] / "data"))
LIGAND_DIR = DATA_DIR / "ligands"
BENCHMARK_DIR = DATA_DIR / "reaction_benchmarks"
EMBEDDING_DIR = DATA_DIR / "embeddings"
