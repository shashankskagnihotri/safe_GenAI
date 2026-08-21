from __future__ import annotations

import argparse
from dataclasses import asdict

from hierasafe_flow.generation.runner import GenerationRunner
from hierasafe_flow.utils.config import apply_overrides, load_config
from hierasafe_flow.utils.io import write_json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run hierarchical vector-field bottleneck generation.")
    parser.add_argument("--config", required=True, help="Experiment YAML path.")
    parser.add_argument("--prompt", default=None, help="Optional prompt overriding config prompt source.")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        help="Override config value with dotted.path=value. Can be repeated.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = apply_overrides(load_config(args.config), args.overrides)
    result = GenerationRunner(config).run(prompt=args.prompt)
    write_json(f"{result.output_dir}/run_result.json", asdict(result))
    print(result.output_dir)


if __name__ == "__main__":
    main()

