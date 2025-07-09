# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Right now the only part of the location that matters is start line and col.
This file provides two utils functions to extract that info from MLIR Location class,
and from raw debug info string that is provided by MLIR runtime.
"""
from typing import Tuple
from ttmlir.ir import Location

UNKNOWN_LOCATION = (-1, -1)


def hash_location(location: Location) -> Tuple[int, int]:
    assert location is not None
    if not hasattr(location, "start_line"):
        print(f"Location does not have start_line: {location}")
        return UNKNOWN_LOCATION
    if not hasattr(location, "start_col"):
        print(f"Location does not have start_col: {location}")
        return UNKNOWN_LOCATION
    return (location.start_line, location.start_col)


def parse_op_location(op_location: str) -> Tuple[int, int]:
    try:
        return tuple(int(x) for x in op_location[:-1].split(":")[-2:])
    except Exception:
        return UNKNOWN_LOCATION
