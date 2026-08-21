"""Model adapters exposing frozen generator vector fields.

Registry exports are resolved lazily so dependency-neutral adapter contracts
can be imported by the condition-cache layer without importing every concrete
adapter (and therefore the cache) recursively.
"""

from typing import Any

__all__ = ["create_adapter", "get_adapter_class", "list_adapters"]


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    from hierasafe_flow.adapters import registry

    return getattr(registry, name)
