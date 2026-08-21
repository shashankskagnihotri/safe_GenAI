from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

from hierasafe_flow.generation.runner import GenerationRunner
from hierasafe_flow.steering.ablations import expand_ablations
from hierasafe_flow.utils.config import apply_overrides, deep_merge, load_config
from hierasafe_flow.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an ablation grid locally.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    base = apply_overrides(load_config(args.config), args.overrides)
    results = []
    for name, config in expand_ablations(base):
        output_dir = Path(config.get("logging", {}).get("output_dir", "outputs")) / name
        config = deep_merge(config, {"logging": {"output_dir": str(output_dir)}})
        result = GenerationRunner(config).run(prompt=args.prompt)
        results.append({"name": name, "result": asdict(result)})
    write_json(Path(base.get("logging", {}).get("output_dir", "outputs")) / "grid_result.json", results)
    print(f"finished {len(results)} runs")


if __name__ == "__main__":
    main()

