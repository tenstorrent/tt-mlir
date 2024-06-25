# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ._C import load_from_path, store

import json


def as_dict(bin):
    return json.loads(bin.as_json())
