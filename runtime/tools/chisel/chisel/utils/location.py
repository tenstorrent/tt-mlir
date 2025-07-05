# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
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
