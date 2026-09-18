# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# RUN: %python -m pytest -q %s

"""Unit tests for semantic profile-region extraction and CSV export."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROFILE_REGIONS_PATH = _REPO_ROOT / "tools" / "ttrt" / "common" / "profile_regions.py"


def _load_profile_regions():
    # Load by path so these tests do not require importing the full ttrt package.
    spec = importlib.util.spec_from_file_location(
        "profile_regions", _PROFILE_REGIONS_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


profile_regions = _load_profile_regions()
PROFILE_REGIONS_COLUMN = profile_regions.PROFILE_REGIONS_COLUMN
annotate_ops_perf_csv = profile_regions.annotate_ops_perf_csv
annotate_ops_perf_rows = profile_regions.annotate_ops_perf_rows
extract_profile_regions = profile_regions.extract_profile_regions
extract_device_op_global_call_count = (
    profile_regions.extract_device_op_global_call_count
)
format_profile_regions = profile_regions.format_profile_regions
profile_regions_from_loc = profile_regions.profile_regions_from_loc


def test_extract_single_fused_region():
    loc = 'loc(fused<{tt.profile.region = "decoder.sdpa"}>["a.mlir":1:1])'
    assert extract_profile_regions(loc) == ["decoder.sdpa"]


def test_extract_quoted_attr_name():
    loc = 'loc(fused<{"tt.profile.region" = "decoder.mlp"}>[...])'
    assert extract_profile_regions(loc) == ["decoder.mlp"]


def test_extract_nested_fused_and_callsite_order():
    loc = (
        'loc(fused<{tt.profile.region = "decoder.mlp"}>'
        '[fused<{tt.profile.region = "decoder.sdpa"}>["a.mlir":1:1], '
        'callsite("b.mlir":2:2 at fused<{tt.profile.region = "decoder.attn"}>'
        '["c.mlir":3:3])])'
    )
    assert extract_profile_regions(loc) == [
        "decoder.mlp",
        "decoder.sdpa",
        "decoder.attn",
    ]


def test_extract_deduplicates_preserving_order():
    loc = (
        'loc(fused<{tt.profile.region = "decoder.sdpa"}>'
        '[fused<{tt.profile.region = "decoder.mlp"}>[...], '
        'fused<{tt.profile.region = "decoder.sdpa"}>[...]])'
    )
    assert extract_profile_regions(loc) == ["decoder.sdpa", "decoder.mlp"]


def test_extract_multiple_regions_on_one_op():
    # Fusion may retain several contributing semantic labels on one emit.
    loc = (
        'loc(fused<{tt.profile.region = "decoder.qkv"}>'
        '[fused<{tt.profile.region = "decoder.sdpa"}>[...]])'
    )
    assert extract_profile_regions(loc) == ["decoder.qkv", "decoder.sdpa"]
    assert format_profile_regions(extract_profile_regions(loc)) == (
        "decoder.qkv;decoder.sdpa"
    )


def test_untagged_location_yields_empty():
    assert extract_profile_regions('loc("file.mlir":10:4)') == []
    assert extract_profile_regions("loc(unknown)") == []
    assert extract_profile_regions(None) == []
    assert extract_profile_regions("") == []
    assert profile_regions_from_loc('loc("file.mlir":10:4)') == ""


def test_extract_device_op_global_call_count_current_tracy_format():
    multiline = (
        '`TT_DNN_DEVICE_OP: "TilizeDeviceOperation", '
        "6579250399439280527, 0, false, 2048 ->"
    )
    single_line_semicolon = (
        '`TT_DNN_DEVICE_OP: "TilizeDeviceOperation", '
        "6579250399439280527, 0, false, 3072`;12167960701"
    )
    # tracy-csvexport sometimes uses a comma after the closing backtick.
    single_line_comma = (
        '`TT_DNN_DEVICE_OP: "TilizeDeviceOperation", '
        "9338796516898729965, 0, false, 3072`,8326920549"
    )
    assert extract_device_op_global_call_count(multiline) == 2048
    assert extract_device_op_global_call_count(single_line_semicolon) == 3072
    assert extract_device_op_global_call_count(single_line_comma) == 3072
    assert extract_device_op_global_call_count("MLIR_OP_LOCATION;loc(unknown)") is None


def test_annotate_rows_preserves_loc():
    rows = [
        {
            "GLOBAL CALL COUNT": "1",
            "LOC": 'loc(fused<{tt.profile.region = "decoder.sdpa"}>[...])',
        },
        {
            "GLOBAL CALL COUNT": "2",
            "LOC": 'loc("untagged.mlir":1:1)',
        },
    ]
    annotated = annotate_ops_perf_rows(rows)
    assert annotated[0]["LOC"] == rows[0]["LOC"]
    assert annotated[0][PROFILE_REGIONS_COLUMN] == "decoder.sdpa"
    assert annotated[1]["LOC"] == rows[1]["LOC"]
    assert annotated[1][PROFILE_REGIONS_COLUMN] == ""


def test_annotate_ops_perf_csv_from_synthetic_tracy_style_input(tmp_path):
    input_csv = tmp_path / "ops_perf_results.csv"
    output_csv = tmp_path / "ops_perf_annotated.csv"
    with open(input_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "GLOBAL CALL COUNT",
                "OP CODE",
                "DEVICE KERNEL DURATION [ns]",
                "LOC",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "GLOBAL CALL COUNT": "7",
                "OP CODE": "ttnn.add",
                "DEVICE KERNEL DURATION [ns]": "1000",
                "LOC": (
                    'loc(fused<{tt.profile.region = "decoder.sdpa"}>' '["x.mlir":1:1])'
                ),
            }
        )
        writer.writerow(
            {
                "GLOBAL CALL COUNT": "8",
                "OP CODE": "ttnn.matmul",
                "DEVICE KERNEL DURATION [ns]": "2000",
                "LOC": 'loc("y.mlir":2:2)',
            }
        )

    annotate_ops_perf_csv(str(input_csv), str(output_csv))

    with open(output_csv, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        assert PROFILE_REGIONS_COLUMN in reader.fieldnames
        assert "LOC" in reader.fieldnames
        rows = list(reader)

    assert rows[0]["LOC"].startswith("loc(fused<{tt.profile.region")
    assert rows[0][PROFILE_REGIONS_COLUMN] == "decoder.sdpa"
    assert rows[1]["LOC"] == 'loc("y.mlir":2:2)'
    assert rows[1][PROFILE_REGIONS_COLUMN] == ""
