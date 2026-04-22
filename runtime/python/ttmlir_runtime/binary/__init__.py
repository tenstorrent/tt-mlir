# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import json
import re

from ttmlir_runtime.runtime._ttmlir_runtime.binary import *


def json_string_as_dict(json_string):
    if json_string == "":
        return {}
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
