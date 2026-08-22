#!/usr/bin/env python3
"""Construct the paper-equation NAG processor without mutating upstream globals.

The pinned NAG release implements four pointwise norms with ``p=2`` although the
paper equations and pseudocode specify L1.  This module obtains the class source
from the already imported pinned module, demands exactly four norm-order sites,
changes only those four integer literals, and compiles a new isolated class.
"""

from __future__ import annotations

import hashlib
import inspect
import re
import textwrap
from types import ModuleType
from typing import Any


EXPECTED_REPLACEMENTS = 4
NORM_ORDER_PATTERN = re.compile(r"\bp\s*=\s*2\b")


def build_paper_l1_processor_class(upstream_module: ModuleType) -> tuple[type, dict[str, Any]]:
    original_class = upstream_module.NAGFluxAttnProcessor2_0
    original_source = textwrap.dedent(inspect.getsource(original_class))
    sites = list(NORM_ORDER_PATTERN.finditer(original_source))
    if len(sites) != EXPECTED_REPLACEMENTS:
        raise RuntimeError(
            "Refusing L1 transformation: expected exactly "
            f"{EXPECTED_REPLACEMENTS} p=2 sites, found {len(sites)}"
        )
    transformed_source = NORM_ORDER_PATTERN.sub("p=1", original_source)
    if transformed_source.count("p=1") < EXPECTED_REPLACEMENTS:
        raise RuntimeError("L1 source transformation postcondition failed")

    namespace = dict(vars(upstream_module))
    namespace["__name__"] = __name__
    exec(compile(transformed_source, "<nag-paper-equation-l1>", "exec"), namespace)
    transformed_class = namespace[original_class.__name__]
    if transformed_class is original_class:
        raise RuntimeError("L1 processor was not isolated from the author class")
    audit = {
        "transformation": "four norm-order keyword literals p=2 -> p=1",
        "replacement_count": len(sites),
        "original_class": f"{original_class.__module__}.{original_class.__qualname__}",
        "original_source_sha256": hashlib.sha256(original_source.encode()).hexdigest(),
        "transformed_source_sha256": hashlib.sha256(transformed_source.encode()).hexdigest(),
    }
    return transformed_class, audit
