# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttrt.runtime._ttmlir_runtime.binary import (
    load_from_path,
    load_binary_from_path,
    load_binary_from_capsule,
    load_system_desc_from_path,
    Flatbuffer,
    GoldenTensor,
)

from . import stats

import json
import re


def as_dict(bin):
    json_txt = bin.as_json()
    # Flatbuffers emits 'nan' and 'inf'
    # But Python's JSON accepts only 'NaN' and 'Infinity' and nothing else
    # We include the comma to avoid replacing 'inf' in contexts like 'info'
    json_txt = re.sub(r"\bnan\b", "NaN", json_txt)
    json_txt = re.sub(r"\binf\b", "Infinity", json_txt)
    return json.loads(json_txt)
