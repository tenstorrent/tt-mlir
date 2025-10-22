# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttrt.runtime._ttmlir_runtime.binary import Binary
import json
import re


def as_dict(bin: Binary):
    return json.loads(bin.as_json())


def _parse_tensor_ref(ref):
    print("REF")
    print(ref)
    if type(ref) == list:
        tensors = (_parse_tensor_ref(r) for r in ref)
        return [t for l in tensors for t in l]

    returns = {
        "shape": ref["desc"]["shape"],
        "data_type": ref["desc"]["layout"]["memory_desc"]["data_type"],
        # "size": ref["desc"]["layout"]["memory_desc"]["size"],
    }
    if "memory_config" in ref["desc"]["layout"]["memory_desc"]:
        returns["memory_config"] = ref["desc"]["layout"]["memory_desc"]["memory_config"]
        if "shard_spec" in ref["desc"]["layout"]["memory_desc"]["memory_config"]:
            returns["core_range_set"] = ref["desc"]["layout"]["memory_desc"][
                "memory_config"
            ]["shard_spec"]["core_range_set"]

    return returns


def _parse_inputs_outputs(operation):
    inputs = []
    outputs = []
    print(operation, type(operation))
    if operation["type_type"] == "GetDeviceOp":
        return inputs, outputs
    for k, v in operation["type"].items():
        if k.startswith("in"):
            print("INNNN")
            print(k, v)
            inputs.extend(_parse_tensor_ref(v))
        elif k.startswith("out"):
            print("OOOUUUT")
            print(k, v)
            # outputs.extend(_parse_tensor_ref(v))
    return inputs, outputs


def _parse_attributes(operation):
    attributes = {}
    return attributes


def collect_op_stats(bin: Binary):
    print("4")
    assert bin.file_identifier == "TTNN", "Only supports TTNN binary files"
    programs = []
    operations = []
    for program_index in range(bin.get_num_programs()):
        print("5")
        json_operations = json.loads(bin.get_program_ops_as_json(program_index))

        pattern = re.compile(r"(?<!^)(?=[A-Z])")
        print("6")

        def to_ttnn_name(name):
            print("ttnn." + pattern.sub("_", name).lower().strip("_op"))
            return "ttnn." + pattern.sub("_", name).lower().strip("_op")

        for operation in json_operations:
            print("7")
            inputs, outputs = _parse_inputs_outputs(operation)
            print(dir(operation))
            print(operation)
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
            print("8")
        programs.append(operations)

    print("9")
    return str(programs)


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
