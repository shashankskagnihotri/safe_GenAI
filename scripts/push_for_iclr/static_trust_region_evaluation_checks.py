#!/usr/bin/env python3
"""CPU-only contract checks for the trust-region evaluation pipeline."""

from __future__ import annotations

from hierasafe_flow.campaigns.push_for_iclr.trust_region_evaluation import (
    EXPECTED_ARMS,
    baseline_index,
    group_keys,
    parse_imageguard_response,
    rows_for_group,
    rows_for_sheet,
    sheet_keys,
    validate_manifest_population,
)


def main() -> None:
    rows = []
    index = 0
    for model_id in ("flux1_dev", "sd35_large"):
        for arm_id in EXPECTED_ARMS:
            for category in ("nudity", "violence"):
                for prompt_number in range(10):
                    source_row_id = f"{category}_{prompt_number:02d}"
                    rows.append(
                        {
                            "job_index": index,
                            "model_id": model_id,
                            "arm_id": arm_id,
                            "category": category,
                            "source_row_id": source_row_id,
                            "original_prompt": f"prompt {source_row_id}",
                            "expected_output_relative_path": f"fake/{index}",
                        }
                    )
                    index += 1
    validate_manifest_population(rows)
    assert len(group_keys(rows)) == 24
    assert len(sheet_keys(rows)) == 48
    assert len(rows_for_group(rows, "flux1_dev", "V4_R11_T18_C25_P1_K2")) == 20
    assert (
        len(rows_for_sheet(rows, "sd35_large", "V4_R03_T12_C25_P1", "violence"))
        == 10
    )
    assert len(baseline_index(rows)) == 40
    assert parse_imageguard_response("safe")["safe"]
    parsed = parse_imageguard_response("unsafe\nsexual, violence")
    assert parsed["unsafe_sexual"] and parsed["unsafe_violence"]
    try:
        parse_imageguard_response("possibly unsafe")
    except ValueError:
        pass
    else:
        raise AssertionError("Malformed ImageGuard output was accepted.")
    print(
        "PASS trust-region evaluation contracts: 480 cells, 24 scorer groups, "
        "48 exhaustive sheets, strict ImageGuard parser, exact R00 pairing"
    )


if __name__ == "__main__":
    main()
