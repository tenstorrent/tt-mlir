# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ._C import (
    Binary,
)
import json
import re


def as_dict(bin: Binary):
    return json.loads(bin.as_json())


def _parse_tensor_ref(ref):
    if type(ref) == list:
        tensors = (_parse_tensor_ref(r) for r in ref)
        return [t for l in tensors for t in l]
    return [
        {
            "shape": ref["desc"]["shape"],
            "data_type": ref["desc"]["layout"]["memory_desc"]["data_type"],
            "memory_space": ref["desc"]["layout"]["memory_desc"]["memory_space"],
            "memory_layout": ref["desc"]["layout"]["memory_desc"]["memory_layout"],
            "core_range_set": ref["desc"]["layout"]["core_range_set"],
        }
    ]


def _parse_inputs_outputs(operation):
    inputs = []
    outputs = []
    if operation["type_type"] == "GetDeviceOp":
        return inputs, outputs
    for k, v in operation["type"].items():
        if k.startswith("in"):
            inputs.extend(_parse_tensor_ref(v))
        elif k.startswith("out"):
            outputs.extend(_parse_tensor_ref(v))
    return inputs, outputs


def _parse_attributes(operation):
    attributes = {}
    return attributes


def collect_op_stats(bin: Binary):
    assert bin.file_identifier == "TTNN", "Only supports TTNN binary files"
    d = as_dict(bin)
    program_index = 0
    operations = []

    pattern = re.compile(r"(?<!^)(?=[A-Z])")

    def to_ttnn_name(name):
        return "ttnn." + pattern.sub("_", name).lower().strip("_op")

    for operation in d["programs"][program_index]["operations"]:
        inputs, outputs = _parse_inputs_outputs(operation)
        operations.append(
            {
                "op_name": to_ttnn_name(operation["type_type"]),
                "framework_op_name": "",
                "dialect_op_name": "",
                "ttir_op_name": "",
                "inputs": inputs,
                "outputs": outputs,
                "attributes": _parse_attributes(operation),
            }
        )

    return operations


def construct_op_stats_json(frontend: str, model: str, bin: Binary):
    op_stats = collect_op_stats(bin)
    return json.dumps(
        {
            "frontend": frontend,
            "model": model,
            "operations": op_stats,
        },
        indent=4,
    )
