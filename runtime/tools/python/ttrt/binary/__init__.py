# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ._C import (
    load_from_path,
    load_binary_from_path,
    load_binary_from_capsule,
    load_system_desc_from_path,
    Flatbuffer,
    GoldenTensor,
    CallbackTag,
)
from . import stats

import json


def as_dict(bin):
    tmp = bin.as_json()
    # Flatbuffers emits 'nan' and 'inf'
    # But Python's JSON accepts only 'NaN' and 'Infinity' and nothing else
    # We include the comma to avoid replacing 'inf' in contexts like 'info'
    tmp = tmp.replace("nan,", "NaN,")
    tmp = tmp.replace("inf,", "Infinity,")
    return json.loads(tmp)
