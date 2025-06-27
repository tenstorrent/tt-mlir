# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import List, Optional, Union, Tuple, Callable, Dict, Any
from ttmlir.ir import *
from ttmlir.dialects import ttir, ttcore, tensor, quant
from ttmlir.passes import GoldenTensor, DataType
import torch
from enum import Enum, auto
import re
from .ccl_golden import *
from .ops import *
from .apis import *
from sphinx.ext.autodoc import FunctionDocumenter

# Alias for operands of ops which can be either BlockArguments, Values, or other
# ops wrapped in OpView or Operation.
Operand = Union[Value, OpView, Operation]

# Convenience alias for shape
Shape = Union[List[int], Tuple[int, ...]]


def get_loc_of_extra_file_callee(id: int = 0) -> Location:
    """When called, this function returns a `Location` referring to first
    callee outside the file of the caller of this function. E.G., if a function
    in `foo.py` called a function in `bar.py` that then called this function,
    the location would be pointing to the call in `foo.py`.

    NOTE: this location is _NOT_ in the form of
    {filename}:{line_number}:{col_number}, but instead in the form:
    {filename}:{line_number}:id({id}), where id is supplied to this function as
    a disambiguator for calls that happen on the same line

    Arguments
    ---------

    id : int
        An optional variable that defaults to 0 to be appended to the location,
        disambiguating calls on the same line.

    Returns
    -------

    A `Location` referring to the first extra file callee of the caller of this function

    """

    stack = inspect.stack()

    # find the innermost frame outside of this file
    caller_filename = stack[1].filename

    while len(stack) > 0 and stack[0].filename == caller_filename:
        stack = stack[1:]

    assert (
        len(stack) > 0
    ), "Top of callstack to builder funcs must be outside the caller's file"

    # FIXME: this should be a `Location.file`, but for some reason it causes
    # strange decomposition inheritance behaviour that breaks using this as
    # a key into the golden map
    return Location.name(f"{stack[0].filename}:{str(stack[0].lineno)}:id({str(id)})")


def get_loc_from_str(loc: Union[str, Location]) -> Location:
    if isinstance(loc, str):
        return Location.name(loc)
    else:
        return loc


@dataclass(frozen=True)
class Golden:
    """
    Dataclass used to store information about golden tensor which will be used
    for comparison with TT device output.

    Each TTIR op should have a matching torch op, and for same inputs, they
    should generate same outputs.
    """

    tensor: torch.Tensor

    # `torch.manual_seed` arg with which tensor was generated. Valid (not None)
    # only for randomly generated tensors, for example args of MLIR function
    # wrapped around user-written op graph. Every other tensor is output of some
    # op from graph.
    seed: Opional[int] = None

    def __repr__(self) -> str:
        s = f"\nRandom seed: {self.seed}" if self.seed is not None else ""
        s += f"\nGolden tensor:\n{self.tensor}"
        return s

    def contiguous(self) -> Golden:
        return Golden(self.tensor.contiguous())


@dataclass
class TypeInfo:
    """Encapsulates type information for quantized tensors.

    Contains both the base data type and quantization parameters (scale and zero point)
    required for quantized operations.

    Attributes:
        dtype: Base PyTorch data type (e.g. torch.float32, torch.qint32).
        scale: Scaling factor for quantization. Required for quantized types.
        zero_point: Zero point offset for quantization. Required for quantized types.
    """

    dtype: torch.dtype
    scale: Optional[float] = None
    zero_point: Optional[int] = None


class GoldenCheckLevel(Enum):
    DISABLED = auto()  # Do not store golden.
    OP_LEVEL = auto()  # Check every single op level goldens
    GRAPH_LEVEL = auto()  # Check graph level goldens only


class TTIRBuilder(TTIRBuilderOps, TTIRBuilderAPIs):
    def __init__(self, ctx: Context, location: Location):
        self._ctx = ctx
        self._loc = location

        self._seed = 0
        # Dictionary to store Golden for each Operand we encounter in MLIR
        # graph.
        self._goldens: Dict[Operand, Golden] = {}

        # global ID of operations
        self._global_id = -1

        # id to golden map
        self.id_golden_map = {}

        # mesh_shape for multi-device
        self.mesh_shape = ()

        # golden check level
        self._golden_check_level = GoldenCheckLevel.OP_LEVEL
