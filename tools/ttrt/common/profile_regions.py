# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""Extract semantic profile regions from MLIR location strings.

Region labels are attached to high-level ops as fused-location metadata:

    loc(fused<{tt.profile.region = "decoder.sdpa"}>[...])

After lowering and fusion, an emitted op may carry an ordered,
de-duplicated set of labels from multiple source regions. This module
parses those labels from the printed location text that already flows
through FlatBuffer ``loc`` / ``loc_info`` and Tracy ``MLIR_OP_LOCATION``.
"""

from __future__ import annotations

import csv
import re
from typing import Iterable, List, Mapping, MutableMapping, Optional, Sequence

PROFILE_REGION_ATTR = "tt.profile.region"
PROFILE_REGIONS_COLUMN = "PROFILE_REGIONS"

# Matches both quoted and bare dictionary keys:
#   tt.profile.region = "decoder.sdpa"
#   "tt.profile.region" = "decoder.sdpa"
_REGION_PATTERN = re.compile(
    r'(?:"tt\.profile\.region"|tt\.profile\.region)\s*=\s*"((?:\\.|[^"\\])*)"'
)

# GLOBAL CALL COUNT is the final comma-separated field before either ``->``
# (multiline op text) or the closing backtick (single-line op text). Earlier
# fields vary with the profiler metadata and cannot be indexed reliably.
# After the closing backtick, tracy-csvexport may emit either ``;`` or ``,``
# before the host timestamp, so do not require a specific trailer.
_DEVICE_OP_CALL_COUNT_PATTERN = re.compile(r"TT_DNN_DEVICE_OP:.*?,\s*(\d+)\s*(?:->|`)")


def extract_profile_regions(loc: Optional[str]) -> List[str]:
    """Return ordered, de-duplicated ``tt.profile.region`` labels from ``loc``.

    Nested fused and callsite locations are handled by scanning the printed
    location left-to-right, which matches MLIR's pre-order print of fused
    metadata before child locations. Operations without the attribute yield
    an empty list.
    """
    if not loc:
        return []

    regions: List[str] = []
    seen = set()
    for match in _REGION_PATTERN.finditer(loc):
        region = bytes(match.group(1), "utf-8").decode("unicode_escape")
        if region in seen:
            continue
        seen.add(region)
        regions.append(region)
    return regions


def extract_device_op_global_call_count(line: str) -> Optional[int]:
    """Extract ``GLOBAL CALL COUNT`` from a Tracy TT_DNN_DEVICE_OP line."""
    match = _DEVICE_OP_CALL_COUNT_PATTERN.search(line)
    return int(match.group(1)) if match else None


def format_profile_regions(regions: Sequence[str]) -> str:
    """Join region labels for the ``PROFILE_REGIONS`` CSV column."""
    return ";".join(regions)


def profile_regions_from_loc(loc: Optional[str]) -> str:
    """Convenience: extract and format regions from a location string."""
    return format_profile_regions(extract_profile_regions(loc))


def annotate_row_with_profile_regions(
    row: MutableMapping[str, str], loc_column: str = "LOC"
) -> None:
    """Set ``PROFILE_REGIONS`` on ``row`` from its ``LOC`` column."""
    row[PROFILE_REGIONS_COLUMN] = profile_regions_from_loc(row.get(loc_column))


def annotate_ops_perf_rows(
    rows: Iterable[Mapping[str, str]], loc_column: str = "LOC"
) -> List[dict]:
    """Return copies of profiler rows with ``PROFILE_REGIONS`` populated."""
    annotated: List[dict] = []
    for row in rows:
        out = dict(row)
        annotate_row_with_profile_regions(out, loc_column=loc_column)
        annotated.append(out)
    return annotated


def annotate_ops_perf_csv(
    input_csv_path: str,
    output_csv_path: str,
    loc_column: str = "LOC",
) -> None:
    """Rewrite an ops-perf CSV, adding ``PROFILE_REGIONS`` from ``LOC``.

    Existing columns, including ``LOC``, are preserved unchanged.
    """
    with open(input_csv_path, mode="r", newline="", encoding="utf-8") as infile:
        reader = csv.DictReader(infile)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {input_csv_path}")
        if loc_column not in reader.fieldnames:
            raise ValueError(f"CSV missing {loc_column!r} column: {input_csv_path}")

        fieldnames = list(reader.fieldnames)
        if PROFILE_REGIONS_COLUMN not in fieldnames:
            fieldnames.append(PROFILE_REGIONS_COLUMN)

        rows = annotate_ops_perf_rows(reader, loc_column=loc_column)

    with open(output_csv_path, mode="w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
