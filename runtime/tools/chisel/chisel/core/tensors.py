from dataclasses import dataclass
from functools import cache
from typing import Any, List, Tuple

import torch
from ttmlir.ir import Operation, BlockArgument, Value, Type

from ttrt.runtime import (
    Tensor,
    create_owned_host_tensor,
    update_tensor_in_pool,
    get_op_output_ref,
    retrieve_tensor_from_pool,
)

from .enums import ExecutionType
from ..utils.location import hash_location, parse_op_location
from ttmlir.ir import Operation

from .enums import ExecutionType



def get_op_outputs(op: Operation):
    outputs = []
    for result in op.results:
        if hasattr(result.type, "shape") and hasattr(result.type, "element_type"):
            outputs.append(result)
    return outputs

def get_op_inputs(op: Operation):
    inputs = []
    for operand in op.operands:
        if hasattr(operand.type, "shape") and hasattr(operand.type, "element_type"):
            inputs.append(operand)
    return inputs

class TensorValue:
    def __init__(self, name: str, data: Any, execution_type: ExecutionType):
        self.name = name
        self.data = data
        self.execution_type = execution_type
        self.tensor_ref: TensorRef | None = None
        self.tensor: Tensor | None = None
        self.execution_data = data

    def set_execution_data(self, data: Any = None):
        if data is None:
            data = self.data
        self.execution_data = data

    def update_tensor(self, program_context):
        if self.tensor_ref is None:
            return
        update_device_tensor(program_context, self.tensor_ref, self.data)

    def __str__(self):
        return f"TensorValue(name={self.name}, data={self.data}, execution_type={self.execution_type}, execution_data={self.execution_data})"

    def __repr__(self):
        return self.__str__()


class TensorPool:
    def __init__(self):
        self.tensors = {}

    def __getitem__(self, key: str):
        return self.tensors[key]

    def __setitem__(self, key: str, value):
        self.tensors[key] = value

    def __contains__(self, key: str):
        return key in self.tensors
