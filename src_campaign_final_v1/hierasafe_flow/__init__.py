"""Hierarchical concept vector-field bottleneck steering."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("hierasafe-flow")
except PackageNotFoundError:
    __version__ = "0.0.0+local"

