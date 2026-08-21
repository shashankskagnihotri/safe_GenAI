#!/usr/bin/env python3
"""Freeze the sealed campaign and write deterministic stage manifests."""

from __future__ import annotations

import argparse
import json

from hierasafe_flow.campaigns.chatgpt_steering import freeze_campaign, load_campaign_spec


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/chatgpt_steering_21_july.yaml")
    args = parser.parse_args()
    frozen = freeze_campaign(load_campaign_spec(args.config))
    print(json.dumps({"matrix_rows": frozen["matrix_rows"], "freeze_sha256": frozen["freeze_sha256"]}, sort_keys=True))


if __name__ == "__main__":
    main()
