# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttmlir
from dataclasses import make_dataclass, is_dataclass, asdict
from collections import defaultdict
from pathlib import Path
from ttmlir.compile_and_run_utils import ModuleDialect

import importlib
import logging
import torch

TTRT_INSTALLED = importlib.util.find_spec("ttrt") is not None


# TODO(ctr-mcampos): update path to be configurable
IR_DUMPS_DIR = 'ir_dumps'
MODEL_EXTENSIONS = ['.ttir', '.mlir', '.ttnn']


def parse_mlir_str(module_str):
    with ttmlir.ir.Context() as ctx:
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
    # This is a dict {name: ..., source: ...}
    for module in cached_modules:
        if module["name"] == at_pass:
            return module["source"]

    logging.error("at_pass=%s not found in Flatbuffer.", at_pass)
    return None


def golden_map_from_flatbuffer(fb_path, program=0):
    # Get the golden_map from flatbuffer corresponding to the Program # provided
    if not TTRT_INSTALLED:
        logging.error(
            "TTRT is not installed in Python Environment, unable to parse Flatbuffer."
        )
        return []
    from ttrt.common.util import Binary, Logger, FileManager

    logger = Logger()
    file_manager = FileManager(logger)

    fbb = Binary(logger, file_manager, fb_path)

    fbb_dict = fbb.fbb_dict

    try:
        debug_info = fbb_dict["programs"][program]["debug_info"]
    except:
        logging.error("Flatbuffer does not contain DebugInfo on Program %d", program)
        return []

    if "golden_info" not in debug_info:
        logging.error("Flatbuffer does not contain Golden Data.")
        return []

    golden_map = debug_info["golden_info"]["golden_map"]

    return golden_map


def to_dataclass(obj: dict, dc_name: str = "tempClass"):
    return make_dataclass(dc_name, ((k, type(v)) for k, v in obj.items()))(**obj)


def add_to_dataclass(dataclass, new_attr_name: str, new_attr_value):
    if not is_dataclass(dataclass):
        return None
    classname = dataclass.__class__.__name__
    dataclass = asdict(dataclass)
    dataclass[new_attr_name] = new_attr_value
    return to_dataclass(dataclass, dc_name=classname)

def to_adapter_collection_format(*objs, **kwargs):
    res = [x if is_dataclass(x) else to_dataclass(x) for x in objs]
    return {"graphCollections": [to_dataclass({ "label": kwargs.get('label', 'Unlabeled collection'), "graphs": res })]}

def to_adapter_format(*objs):
    res = [x if is_dataclass(x) else to_dataclass(x) for x in objs]
    return {"graphs": res}


# TODO(odjuricic): Better way would be to change KeyValue class to support editable instead.
def make_editable_kv(kv, editable):
    obj = asdict(kv)
    obj["editable"] = editable
    return to_dataclass(obj, "KeyValue")


def is_nested_module(op):
    # Check for ttcore.device_module or builtin.module operations
    return (
        op.operation.name == "ttcore.device_module"
        or op.operation.name == "builtin.module"
    )


def needs_stablehlo_pass(module_path: str) -> bool:
    with open(module_path, "r") as model_file:
        module_str = model_file.read()

    module_dialect = ModuleDialect.detect(module_str)

    return module_dialect == ModuleDialect.STABLE_HLO

def get_collection_path(model_path: str):
    resolved_model_path = Path(model_path).resolve()

    # If the path is adirectory and has an "extension", simply return it.
    if os.is_dir(resolved_model_path) and MODEL_EXTENSIONS.count(resolved_model_path.suffix) > 0:
        return resolved_model_path

    resolved_ir_dir = Path(IR_DUMPS_DIR).resolve()

    for parent in resolved_model_path.parents:
        if parent == resolved_ir_dir:
            # Don't walk up more than the IR directory.
            break

        if MODEL_EXTENSIONS.count(parent.suffix) > 0:
            # Resolve to the closest directory with an "extension".
            return parent

    # Resolve to the file itself.
    return resolved_model_path

def get_collection_label(model_path: str):
    collection_path = get_collection_path(model_path=model_path)

    return collection_path.name
