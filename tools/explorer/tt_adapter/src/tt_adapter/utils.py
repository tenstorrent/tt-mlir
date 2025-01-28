# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttmlir
from dataclasses import make_dataclass, is_dataclass, asdict

import importlib
import logging
import torch

TTRT_INSTALLED = importlib.util.find_spec("ttrt") is not None


def parse_mlir_str(module_str):
    with ttmlir.ir.Context() as ctx:
        ttmlir.dialects.ttir.register_dialect(ctx)
        ttmlir.dialects.tt.register_dialect(ctx)
        ttmlir.dialects.ttnn.register_dialect(ctx)
        module = ttmlir.ir.Module.parse(module_str, ctx)
        return module


def parse_flatbuffer_file(fb_path, at_pass=None, program=0):
    if not TTRT_INSTALLED:
        logging.error(
            "TTRT is not installed in Python Environment, unable to parse Flatbuffer"
        )
        return None

    from ttrt.common.util import Binary, Logger, FileManager

    logger = Logger()
    file_manager = FileManager(logger)

    fbb = Binary(logger, file_manager, fb_path)
    # This will load a Binary that we will parse to see if the correct attributes are present
    # Get the Flatbuffer as a Dict
    fbb_dict = fbb.fbb_dict
    # Get the MLIR_stages for the first program. If multiple other programs are needed in the future, parse as such.
    try:
        debug_info = fbb_dict["programs"][program]["debug_info"]
    except:
        logging.error("Flatbuffer does not contain DebugInfo on Program %d", program)
        return None

    if "mlir_stages" not in debug_info:
        # MLIR Stages not present
        logging.error(
            "Flatbuffer does not contain Cached Module, invalid for explorer."
        )
        return None

    cached_modules = debug_info["mlir_stages"]
    for (pass_name, module) in cached_modules:
        if pass_name == at_pass:
            return module

    logging.error("at_pass=%s not found in Flatbuffer.", at_pass)
    return None


def golden_map_from_flatbuffer(fb_path, program=0):
    # Get the golden_map from flatbuffer corresponding to the Program # provided
    if not TTRT_INSTALLED:
        logging.error(
            "TTRT is not installed in Python Environment, unable to parse Flatbuffer."
        )
        return None
    from ttrt.common.util import Binary, Logger, FileManager

    logger = Logger()
    file_manager = FileManager(logger)

    fbb = Binary(logger, file_manager, fb_path)

    fbb_dict = fbb.fbb_dict

    try:
        debug_info = fbb_dict["programs"][program]["debug_info"]
    except:
        logging.error("Flatbuffer does not contain DebugInfo on Program %d", program)
        return None

    if "golden_info" not in debug_info:
        logging.error("Flatbuffer does not contain Golden Data.")
        return None

    golden_map = debug_info["golden_info"]["golden_map"]
    rendered_golden_map = {}

    # Create a np array from the data:
    data_tensor = torch.Tensor(data["data"])

    for entry in golden_map:
        data = entry["value"]
        tensor = ttmlir.passes.create_golden_tensor(
            data["name"],
            data["shape"],
            data["strides"],
            ttmlir.passes.lookup_dtype(data["dtype"]),
            data_tensor.data_ptr,
        )
        rendered_golden_map[entry["key"]] = tensor

    return rendered_golden_map


def to_dataclass(obj: dict, dc_name: str = "tempClass"):
    return make_dataclass(dc_name, ((k, type(v)) for k, v in obj.items()))(**obj)


def add_to_dataclass(dataclass, new_attr_name: str, new_attr_value):
    if not is_dataclass(dataclass):
        return None
    classname = dataclass.__class__.__name__
    dataclass = asdict(dataclass)
    dataclass[new_attr_name] = new_attr_value
    return to_dataclass(dataclass, dc_name=classname)


def to_adapter_format(*objs):
    res = [x if is_dataclass(x) else to_dataclass(x) for x in objs]
    return {"graphs": res}


# TODO(odjuricic): Better way would be to change KeyValue class to support editable instead.
def make_editable_kv(kv, editable):
    obj = asdict(kv)
    obj["editable"] = editable
    return to_dataclass(obj, "KeyValue")
