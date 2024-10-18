# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from model_explorer import graph_builder
from dataclasses import make_dataclass, asdict
import pathlib


def get_loc_str(loc):
    try:
        res = str(loc).split('"')[1]
    except:
        res = "unknown"
    return res


def array_ref_repr(array_ref):
    return str(list(array_ref))


def make_editable_kv(kv, editable):
    obj = asdict(kv)
    obj["editable"] = editable
    return make_dataclass("KeyValue", ((k, type(v)) for k, v in obj.items()))(**obj)


def to_dataclass(obj: dict, dc_name: str = "tempClass"):
    return make_dataclass(dc_name, ((k, type(v)) for k, v in obj.items()))(**obj)


def to_adapter_format(obj: dict):
    return {"graphs": [to_dataclass(obj)]}


def is_valid_path(path: str):
    try:
        path = pathlib.Path(path)
        return True
    except:
        return False


def get_enum_options(enum_type):
    return [str(e) for e in enum_type]
