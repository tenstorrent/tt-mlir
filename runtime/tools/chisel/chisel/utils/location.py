# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from typing import Tuple
from ttmlir.ir import Location

UNKNOWN_LOCATION = (-1, -1)


def hash_location(location: Location) -> Tuple[int, int]:
    assert location is not None
    return (location.start_line, location.start_col)


def parse_op_location(op_location: str) -> Tuple[int, int]:
    try:
        return tuple(int(x) for x in op_location[:-1].split(":")[-2:])
    except Exception:
        return UNKNOWN_LOCATION
