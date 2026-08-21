from __future__ import annotations

import argparse
from dataclasses import asdict

from hierasafe_flow.generation.runner import GenerationRunner
from hierasafe_flow.utils.config import apply_overrides, load_config
from hierasafe_flow.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run dummy-adapter smoke generation.")
    parser.add_argument("--config", default="configs/experiments/smoke_t2i.yaml")
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = apply_overrides(load_config(args.config), args.overrides)
    if config.get("model", {}).get("adapter") != "dummy":
        raise ValueError("Smoke tests must use the dummy adapter to avoid model downloads.")
    result = GenerationRunner(config).run(prompt=args.prompt)
    write_json(f"{result.output_dir}/smoke_result.json", asdict(result))
    print(result.output_dir)


if __name__ == "__main__":
    main()

