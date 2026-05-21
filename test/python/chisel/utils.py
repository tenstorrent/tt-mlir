# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import json
import re


def json_string_as_dict(s: str) -> dict:
    """Parse a flatbuffer-emitted JSON string, normalizing nan/inf for json.loads."""
    if not s:
        return {}
    s = re.sub(r"\bnan\b", "NaN", s)
    s = re.sub(r"\binf\b", "Infinity", s)
    return json.loads(s)


def iterate_programs(binary):
    """Yield (index, name) for each program in the binary."""
    for i in range(binary.get_num_programs()):
        yield i, binary.get_program_name(i)
