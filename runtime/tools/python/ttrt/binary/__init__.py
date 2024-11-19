# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ._C import (
    load_from_path,
    load_binary_from_path,
    load_binary_from_capsule,
    load_system_desc_from_path,
    Flatbuffer,
)
from . import stats

import json


def as_dict(bin):
    return json.loads(bin.as_json())
