"""
ADRiver-4D — Advection–Diffusion–Reaction Informed 4D Point Cloud Encoder.

`ADRiver4DEncoder` loads Pointnet2/Mamba deps; import lazily to keep `adr_operator` usable alone.
"""

from .adr_operator import ADRiverDynamics, ADRiverRefinement

__version__ = "0.2.0"

__all__ = [
    "ADRiverDynamics",
    "ADRiverRefinement",
    "ADRiver4DEncoder",
    "__version__",
]


def __getattr__(name):
    if name == "ADRiver4DEncoder":
        from .encoder import ADRiver4DEncoder as _E

        return _E
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
