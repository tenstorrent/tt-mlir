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


def json_string_as_dict(json_string):
    if json_string == "":
        return {}

    # Flatbuffers emits 'nan' and 'inf'
    # But Python's JSON accepts only 'NaN' and 'Infinity' and nothing else
    # We include the comma to avoid replacing 'inf' in contexts like 'info'
    json_string = re.sub(r"\bnan\b", "NaN", json_string)
    json_string = re.sub(r"\binf\b", "Infinity", json_string)
    return json.loads(json_string)


def fbb_as_dict(bin):
    return json_string_as_dict(bin.as_json())


def system_desc_as_dict(bin):
    return json_string_as_dict(bin.get_system_desc_as_json())


def program_ops_as_dict(bin, index):
    return json_string_as_dict(bin.get_program_ops_as_json(index))


def program_inputs_as_dict(bin, index):
    return json_string_as_dict(bin.get_program_inputs_as_json(index))


def program_outputs_as_dict(bin, index):
    return json_string_as_dict(bin.get_program_outputs_as_json(index))


def mlir_as_dict(bin):
    return json_string_as_dict(bin.get_mlir_as_json())
