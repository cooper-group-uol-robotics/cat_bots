"""Registry of measured reaction grids that act as experiment oracles."""

from functools import partial

from catbots.benchmarks.cernak import load_cernak_cn
from catbots.benchmarks.doyle import load_doyle_arylation
from catbots.benchmarks.perera import load_perera_suzuki
from catbots.benchmarks.shields import load_shields_arylation
from catbots.oracle import ReactionOracle

_REGISTRY = {
    "shields_arylation": load_shields_arylation,
    "doyle_arylation": load_doyle_arylation,
    "perera_suzuki": partial(load_perera_suzuki, subset="full"),
    "perera_suzuki_edbo": partial(load_perera_suzuki, subset="edbo"),
    "cernak_cn": load_cernak_cn,
}


def list_benchmarks() -> list[str]:
    return list(_REGISTRY)


def load_benchmark(name: str, **kwargs) -> ReactionOracle:
    if name not in _REGISTRY:
        raise KeyError(f"unknown benchmark {name!r}; available: {list_benchmarks()}")
    return _REGISTRY[name](**kwargs)


__all__ = [
    "list_benchmarks",
    "load_benchmark",
    "load_cernak_cn",
    "load_doyle_arylation",
    "load_perera_suzuki",
    "load_shields_arylation",
]
