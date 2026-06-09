from __future__ import annotations

import argparse
import json

from hierasafe_flow.adapters.registry import create_adapter, list_adapters
from hierasafe_flow.utils.config import apply_overrides, load_config
from hierasafe_flow.utils.device import resolve_device, resolve_dtype


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect adapter mapping and optional runtime components.")
    parser.add_argument("--config", default=None, help="Config containing a model section.")
    parser.add_argument("--adapter", default=None, help="Adapter name when no config is supplied.")
    parser.add_argument("--model-id", default=None, help="Model id when no config is supplied.")
    parser.add_argument("--load", action="store_true", help="Actually load diffusers pipeline for validation.")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--list", action="store_true", help="List registered adapters.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.list:
        print(json.dumps(list_adapters(), indent=2))
        return

    if args.config:
        config = apply_overrides(load_config(args.config), args.overrides)
        model_config = config.get("model", {})
        runtime = config.get("runtime", {})
    else:
        model_config = {"adapter": args.adapter, "model_id": args.model_id}
        runtime = {"device": "cpu", "dtype": "float32"}

    adapter = create_adapter(
        model_config,
        device=resolve_device(str(runtime.get("device", "cpu"))),
        dtype=resolve_dtype(str(runtime.get("dtype", model_config.get("torch_dtype", "float32")))),
    )
    if args.load:
        adapter.load()
    print(json.dumps(adapter.inspect(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

