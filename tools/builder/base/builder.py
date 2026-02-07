# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations
import inspect
from typing import List, Optional, Union, Tuple, Callable, Dict, Any
import torch
from enum import Enum, auto
import re
from collections import OrderedDict

from ttmlir.ir import *
from ttmlir.dialects import tensor, quant, func, ttir, ttcore, stablehlo, ttnn, debug
from ttmlir.passes import GoldenTensor, DataType
from golden import GoldenMapTensor, get_golden_function

from builder.base.builder_utils import (
    process_multi_return_result,
    TypeInfo,
    tag,
    parse,
    split,
)


class BuilderMeta(type):
    """Metaclass that automatically registers op builder, parser, and split methods.

    When a new ``Builder`` subclass is defined, this metaclass scans every method
    for the ``_tag``, ``_parse``, and ``_split`` attributes (set by the
    :func:`~builder.base.builder_utils.tag`,
    :func:`~builder.base.builder_utils.parse`, and
    :func:`~builder.base.builder_utils.split` decorators, respectively) and
    populates the class-level dispatch maps so that the correct method is called
    for each ``OpView`` type at build / parse / split time.
    """

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)
        cls.build_opview_to_builder_map()
        cls.build_opview_to_parser_map()
        cls.build_opview_to_split(map)
        return cls


class Builder(metaclass=BuilderMeta):
    """Base class for MLIR op builders with golden-tensor tracking.

    ``Builder`` provides the shared infrastructure used by every dialect-specific
    builder (``TTIRBuilder``, ``StableHLOBuilder``, ``TTNNBuilder``, …).  Its
    responsibilities include:

    * **Op dispatch** – class-level maps from ``OpView`` types to the methods
      that build, parse, or split those ops.
    * **Golden management** – generating, storing, and retrieving reference
      ("golden") tensors so that compiled outputs can be verified against
      known-good values.
    * **Module construction helpers** – creating ``func.FuncOp`` wrappers,
      ``call`` operations, and device / CPU module scaffolding.
    * **Type conversion** – bidirectional mapping between ``torch.dtype`` /
      ``TypeInfo`` and MLIR element types (including quantised types).

    Subclasses must implement :meth:`_get_empty_op` (and optionally
    :meth:`create_tensor_encoding`) to provide dialect-specific empty-tensor
    creation.

    Parameters
    ----------
    ctx : Context
        The MLIR context in which all IR is created.
    location : Location
        Default location attached to newly created ops.
    mesh_name : Union[List[str], str]
        Symbolic name(s) for the device mesh(es).
    mesh_dict : Union[List[OrderedDict[str, int]], OrderedDict[str, int]]
        Shape of each mesh, keyed by axis name (e.g. ``OrderedDict([("x", 1), ("y", 1)])``).
    disable_golden_check : bool
        When ``True``, skip golden-tensor generation and comparison entirely.
    """

    opview_to_builder_map: Dict[OpView, Callable] = {}
    """Maps each ``OpView`` type to the builder method that constructs it."""

    opview_to_parser_map: Dict[OpView, Callable] = {}
    """Maps each ``OpView`` type to the parser method that reconstructs it."""

    opview_to_split_map: Dict[OpView, Callable] = {}
    """Maps each ``OpView`` type to the split method that isolates it."""

    # ----- Methods -----

    def __init__(
        self,
        ctx: Context,
        location: Location,
        mesh_name: Union[List[str], str] = "mesh",
        mesh_dict: Union[
            List[OrderedDict[str, int]], OrderedDict[str, int]
        ] = OrderedDict([("x", 1), ("y", 1)]),
        disable_golden_check: bool = False,
    ):
        self._ctx = ctx
        self._loc = location
        self._global_id = -1
        self._disable_golden_check = disable_golden_check
        self._force_graph_level_check = False

        # Keep a list of inputs and outputs in order so we know how to store them in golden map.
        # ordered dict determines program order when comparing goldens during runtime
        # func_op: [[ordered_inputs], [ordered_outputs]]
        self._func_ops_generated: Dict[func.FuncOp, List[List[Operand]]] = {}

        # Explicity set goldens to store. If empty, store all goldens.
        self._goldens_to_store: List[Operand] = []

        # Map from operand to its golden tensor.
        self._goldens: Dict[Operand, GoldenMapTensor] = {}

        # Map from operand to its location string.
        self._operand_to_loc: Dict[Operand, str] = {}

        # Map from location string to the operand at that location.
        self._loc_to_operand: Dict[str, Operand] = {}

        # List of op locations to bypass golden comparison.
        self._bypass_ops: List[str] = []

        # Set torch seed for reproducibility.
        torch.manual_seed(0)

        if not isinstance(mesh_name, List):
            mesh_name = [mesh_name]
        if not isinstance(mesh_dict, List):
            mesh_dict = [mesh_dict]
        if len(mesh_name) != len(mesh_dict):
            raise ValueError(
                f"mesh_name length {len(mesh_name)} must match mesh_dict length {len(mesh_dict)}"
            )
        self._meshes = {}
        for name, mesh in zip(mesh_name, mesh_dict):
            self._meshes[name] = mesh

        self._mesh_shape = tuple(mesh_dict[0].values())

        # Internal values to keep track
        self._root_module_insertion_point = None
        self._current_module_insertion_point = None
        self._cpu_module_insertion_point = None
        self._hoisted_cpu_functions: List[str] = []
        self._nested_funcs: List[str] = []
        self._func_name_to_op: Dict[str, func.FuncOp] = {}

    # ----- Class helper methods -----

    @classmethod
    def build_opview_to_builder_map(cls):
        """Scan class methods for ``@tag`` decorators and register them in
        :attr:`opview_to_builder_map`."""
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            func = attr

            if callable(attr) and hasattr(func, "_tag"):
                cls.opview_to_builder_map[func._tag] = attr

    @classmethod
    def build_opview_to_parser_map(cls):
        """Scan class methods for ``@parse`` decorators and register them in
        :attr:`opview_to_parser_map`."""
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            func = attr

            if callable(attr) and hasattr(func, "_parse"):
                cls.opview_to_parser_map[func._parse] = attr

    @classmethod
    def build_opview_to_split(cls, map):
        """Scan class methods for ``@split`` decorators and register them in
        :attr:`opview_to_split_map`."""
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            func = attr

            if callable(attr) and hasattr(func, "_split"):
                cls.opview_to_split_map[func._split] = attr

    def get_opview_from_method(self, method: func) -> OpView:
        """Return the ``OpView`` type associated with a ``@tag``-decorated builder method."""
        return getattr(method, "_tag", None)

    def get_opview_from_parser(self, parser: func) -> OpView:
        """Return the ``OpView`` type associated with a ``@parse``-decorated parser method."""
        return getattr(parser, "_parse", None)

    def get_opview_from_split(self, split: func) -> OpView:
        """Return the ``OpView`` type associated with a ``@split``-decorated split method."""
        return getattr(split, "_split", None)

    def get_builder_from_opview(self, opview: OpView) -> Callable:
        """Look up the builder method registered for *opview*.

        Raises
        ------
        AssertionError
            If no builder has been registered for the given ``OpView``.
        """
        if opview not in self.opview_to_builder_map:
            assert False, f"No builder found for opview {opview}"
        return self.opview_to_builder_map.get(opview)

    def get_parser_from_opview(self, opview: OpView) -> Callable:
        """Look up the parser method registered for *opview*.

        Raises
        ------
        AssertionError
            If no parser has been registered for the given ``OpView``.
        """
        if opview not in self.opview_to_parser_map:
            assert False, f"No parser found for opview {opview}"
        return self.opview_to_parser_map.get(opview)

    def get_split_from_opview(self, opview: OpView) -> Callable:
        """Look up the split method registered for *opview*.

        Raises
        ------
        AssertionError
            If no split method has been registered for the given ``OpView``.
        """
        if opview not in self.opview_to_split_map:
            assert False, f"No split function found for opview {opview}"
        return self.opview_to_split_map.get(opview)

    # ----- Public methods -----

    @property
    def context(self) -> Context:
        """The MLIR ``Context`` used by this builder."""
        return self._ctx

    @property
    def location(self) -> Location:
        """The default ``Location`` attached to newly created operations."""
        return self._loc

    @property
    def mesh_shape(self) -> Tuple[int, int]:
        """Shape of the first device mesh as ``(rows, cols)``."""
        return self._mesh_shape

    @property
    def golden_map(
        self,
    ) -> Tuple[
        Dict[int, Dict[str, Dict[int, GoldenMapTensor]]],
        Dict[str, Dict[int, GoldenMapTensor]],
    ]:
        """Collect all golden tensors into two dictionaries for runtime verification.

        The first dictionary contains input/output goldens keyed by
        ``{program_index: {loc_str: {device_id: GoldenMapTensor}}}``.
        The second contains intermediate-op goldens keyed by
        ``{loc_str: {device_id: GoldenMapTensor}}``.

        If :attr:`_disable_golden_check` is set, both dictionaries are empty.

        Returns
        -------
        Tuple[Dict[int, Dict[str, Dict[int, GoldenMapTensor]]], Dict[str, Dict[int, GoldenMapTensor]]]
            ``(input_output_goldens, intermediate_goldens)``
        """
        # { program_index: {loc: {device_id: GoldenMapTensor} } }
        input_output_golden_info: Dict[int, Dict[str, Dict[int, GoldenMapTensor]]] = {}
        intermediate_golden_info: Dict[str, Dict[int, GoldenMapTensor]] = {}

        if self._disable_golden_check:
            return input_output_golden_info, intermediate_golden_info

        # If no specific golden is marked to be stored, store all goldens.
        if len(self._goldens_to_store) == 0:
            self._goldens_to_store = list(self._goldens.keys())

        # Iterate through all functions generated to collect their inputs and outputs.
        programs = []
        for func_op, ordered_values in self._func_ops_generated.items():
            if (
                func_op.name.value in self._hoisted_cpu_functions
                or func_op.name.value in self._nested_funcs
            ):
                continue

            programs.append(ordered_values)

        for program_index, ordered_values in enumerate(programs):
            input_output_golden_info[program_index] = {}
            ordered_inputs, ordered_outputs = ordered_values

            # Always store inputs into golden map.
            for index, input in enumerate(ordered_inputs):
                loc = f"input_{index}"
                input_output_golden_info[program_index][loc] = self._get_golden_tensor(
                    input
                )

            # Store outputs into golden map if they are marked to be stored.
            for index, output in enumerate(ordered_outputs):
                if output not in self._goldens_to_store:
                    continue

                loc = f"output_{index}"
                input_output_golden_info[program_index][loc] = self._get_golden_tensor(
                    output
                )

        if self._force_graph_level_check:
            return input_output_golden_info, intermediate_golden_info

        # Store other operands into golden map if they are marked to be stored.
        for operand, golden_map_tensor in self._goldens.items():
            if operand not in self._goldens_to_store:
                continue

            if not isinstance(operand, OpResult):
                continue

            loc = self._operand_to_loc.get(operand, None)
            self._loc_to_operand[loc] = operand
            intermediate_golden_info[loc] = golden_map_tensor

        return input_output_golden_info, intermediate_golden_info

    def get_shape(self, input: Operand) -> Shape:
        """Return the tensor shape of *input*.

        Parameters
        ----------
        input : Operand
            An MLIR operand whose type is ``RankedTensorType``.

        Returns
        -------
        Shape
            The shape (list of dimension sizes) of the tensor.
        """
        return self._get_type(input).shape

    def get_type(self, input: Operand) -> Type:
        """Return the element type of *input*.

        Parameters
        ----------
        input : Operand
            An MLIR operand whose type is ``RankedTensorType``.

        Returns
        -------
        Type
            The MLIR element type (e.g. ``F32Type``, ``IntegerType``).
        """
        return self._get_type(input).element_type

    def set_goldens(
        self,
        inputs: Dict[Operand, Union[Callable, torch.tensor, Dict[int : torch.tensor]]],
        outputs: Dict[Operand, Union[torch.tensor, Dict[int : torch.tensor]]] = None,
        set_all_outputs: bool = True,
    ):
        """Register golden reference tensors for input (and optionally output) operands.

        Each value may be a ``torch.Tensor``, a ``Dict[int, torch.Tensor]``
        mapping device-ids to per-device shards, or a ``Callable`` that accepts
        a shape and returns a ``torch.Tensor``.

        Parameters
        ----------
        inputs : Dict[Operand, Union[Callable, torch.Tensor, Dict[int, torch.Tensor]]]
            Golden tensors for input operands.
        outputs : Dict[Operand, Union[torch.Tensor, Dict[int, torch.Tensor]]], optional
            Golden tensors for output operands.
        set_all_outputs : bool
            When ``True`` (default), all *outputs* operands are automatically
            marked for golden comparison via :meth:`set_goldens_to_check`.
        """
        self._set_goldens(self._create_builder_golden_from_torch_tensor(inputs))

        if outputs != None:
            self._set_goldens(self._create_builder_golden_from_torch_tensor(outputs))
            if set_all_outputs:
                self.set_goldens_to_check(outputs.keys())

    def set_goldens_from_builder_tensor(
        self,
        inputs: Dict[Operand, GoldenMapTensor],
        outputs: Dict[Operand, GoldenMapTensor] = None,
    ):
        """Register pre-built :class:`GoldenMapTensor` objects as goldens.

        Unlike :meth:`set_goldens`, the tensors here are already wrapped in
        ``GoldenMapTensor`` and need no further conversion.

        Parameters
        ----------
        inputs : Dict[Operand, GoldenMapTensor]
            Golden map tensors for input operands.
        outputs : Dict[Operand, GoldenMapTensor], optional
            Golden map tensors for output operands.
        """
        self._set_goldens(inputs)

        if outputs != None:
            self.set_goldens_to_check(outputs.keys())
            self._set_goldens(outputs)

    def set_operand_goldens(
        self, operands: Dict[Operand, Union[torch.tensor, Dict[int : torch.tensor]]]
    ):
        """Register golden tensors for arbitrary operands and mark them for checking.

        This is a convenience wrapper that combines :meth:`set_goldens` and
        :meth:`set_goldens_to_check` for intermediate operands (not limited to
        inputs/outputs).

        Parameters
        ----------
        operands : Dict[Operand, Union[torch.Tensor, Dict[int, torch.Tensor]]]
            Mapping from operands to their golden tensors.
        """
        self._set_goldens(self._create_builder_golden_from_torch_tensor(operands))
        self.set_goldens_to_check(operands.keys())

    def set_goldens_to_check(self, operands: List[Operand], override: bool = False):
        """Mark *operands* for inclusion in the golden-comparison pass.

        By default the new operands are appended to the existing list.  Pass
        ``override=True`` to replace the list entirely.

        Parameters
        ----------
        operands : List[Operand]
            Operands whose golden tensors should be verified at runtime.
        override : bool
            If ``True``, replace the existing list instead of extending it.
        """
        if override:
            self._goldens_to_store = operands
        else:
            self._goldens_to_store.extend(operands)

    def set_graph_level_check(self, check: bool):
        """Enable or disable graph-level-only golden comparison.

        When enabled, only program inputs and outputs are compared—intermediate
        op goldens are skipped.

        Parameters
        ----------
        check : bool
            ``True`` to restrict comparison to graph-level I/O only.
        """
        self._force_graph_level_check = check

    def bypass(self, operand: Operand):
        """Exclude *operand*'s producing op from golden comparison at runtime.

        Parameters
        ----------
        operand : Operand
            The ``OpResult`` whose producing op should be bypassed.

        Raises
        ------
        TypeError
            If *operand* is a ``BlockArgument`` (block arguments have no
            producing op to bypass).
        """
        if isinstance(operand, BlockArgument):
            raise TypeError("Cannot bypass BlockArgument")

        loc = str(operand.owner.location)
        self._bypass_ops.append(loc)

    def set_arg_attribute(
        self, operand: Operand, new_attr_name: str, new_attr: Attribute
    ):
        """Attach or replace a named attribute on a function argument.

        This modifies the ``arg_attrs`` array on the enclosing ``func.FuncOp``
        so that the argument at ``operand.arg_number`` carries the new attribute.

        Parameters
        ----------
        operand : Operand
            A ``BlockArgument`` belonging to a ``func.FuncOp``.
        new_attr_name : str
            Name of the attribute to set (e.g. ``"tt.layout"``).
        new_attr : Attribute
            The MLIR ``Attribute`` value to attach.
        """
        func_op = operand.owner.owner

        arg_attr_list = func_op.arg_attrs
        new_arg_attr_list = []
        for arg_number, arg_attrs in enumerate(arg_attr_list):
            if arg_number == operand.arg_number:
                new_arg_attr = {}
                for attr in arg_attrs:
                    new_arg_attr[attr.name.value] = attr
                new_arg_attr[new_attr_name] = new_attr
                new_arg_attr_list.append(DictAttr.get(new_arg_attr))
            else:
                new_arg_attr_list.append(arg_attrs)

        func_op.arg_attrs = ArrayAttr.get(new_arg_attr_list)

    # ----- Private methods -----

    def _get_output_shape_and_type(
        self,
        organize_golden_args: Callable,
        inputs: List[Operand],
        op_function: Callable,
        golden_kwargs: dict = {},
    ):
        """Infer the output shape and dtype by running the golden function.

        Because TTIR ops do not carry MLIR shape-inference traits, the builder
        runs the Python golden function eagerly to determine what shape and
        dtype the output tensor should have.

        Parameters
        ----------
        organize_golden_args : Callable
            Function that maps a list of ``Operand`` to the positional
            arguments expected by the golden function.
        inputs : List[Operand]
            Input operands (may be empty for ops like ``ttir.zeros``).
        op_function : Callable
            The MLIR op constructor (e.g. ``ttir.AddOp``).
        golden_kwargs : dict
            Extra keyword arguments forwarded to the golden function.

        Returns
        -------
        Optional[Tuple[Shape, torch.dtype]]
            ``(shape, dtype)`` of the golden output, or ``None`` if no golden
            function is registered for *op_function*.
        """
        op_golden_function = get_golden_function(op_function, **golden_kwargs)
        if op_golden_function is None:
            return

        # If the op has no input, just call golden function with kwargs (eg ttir.zeros).
        if len(inputs) == 0:
            golden_output = op_golden_function(**golden_kwargs)
        else:
            golden_output = op_golden_function(
                *(organize_golden_args(inputs)), **golden_kwargs
            )

        return golden_output.shape, golden_output.dtype

    def _get_datatype_from_torch_dtype(self, dtype: torch.dtype) -> DataType:
        """Convert a ``torch.dtype`` to the flatbuffer ``DataType`` enum used by the runtime."""
        match dtype:
            case torch.float16:
                return DataType.Float16
            case torch.bfloat16:
                return DataType.BFloat16
            case torch.uint8:
                return DataType.UInt8
            case torch.uint16:
                return DataType.UInt16
            case torch.uint32:
                return DataType.UInt32
            case torch.int32 | torch.qint32:
                return DataType.Int32
            case torch.int64:
                return DataType.Int32
            case torch.float32 | None:
                return DataType.Float32

    def _get_type(self, input: Operand) -> RankedTensorType:
        """Return the ``RankedTensorType`` of *input* (shape + element type + encoding)."""
        return input.type

    def _get_type_from_torch_dtype(
        self,
        dtype: Union[torch.dtype, TypeInfo],
        scale: Optional[float] = None,
        zero_point: Optional[float] = None,
    ) -> Type:
        """Convert a ``torch.dtype`` (or ``TypeInfo``) to the corresponding MLIR ``Type``.

        For quantised types (``torch.qint32``, ``torch.qint8``, ``torch.quint8``)
        a :class:`quant.UniformQuantizedType` is returned using the scale and
        zero-point carried by *dtype* (which must be a ``TypeInfo`` instance).

        Parameters
        ----------
        dtype : Union[torch.dtype, TypeInfo]
            The source dtype.  If *scale* and *zero_point* are passed
            explicitly they are wrapped into a ``TypeInfo`` automatically.
        scale : float, optional
            Quantisation scale (shortcut; overrides ``TypeInfo.scale``).
        zero_point : float, optional
            Quantisation zero-point (shortcut; overrides ``TypeInfo.zero_point``).

        Returns
        -------
        Type
            The MLIR element type.

        Raises
        ------
        TypeError
            If *dtype* is not a recognised ``torch.dtype``.
        ValueError
            If a quantised dtype is given without the required scale/zero-point.
        """
        if scale is not None and zero_point is not None:
            dtype = TypeInfo(dtype=dtype, scale=scale, zero_point=zero_point)
        base_dtype = dtype.dtype if isinstance(dtype, TypeInfo) else dtype

        match base_dtype:
            case torch.bfloat16:
                return BF16Type.get(self._ctx)
            case torch.float16:
                return F16Type.get(self._ctx)
            case torch.float32:
                return F32Type.get(self._ctx)
            case torch.float64:
                return F64Type.get(self._ctx)
            case torch.int8:
                return IntegerType.get_signless(8, self._ctx)
            case torch.int16:
                return IntegerType.get_signless(16, self._ctx)
            case torch.int32:
                return IntegerType.get_signless(32, self._ctx)
            case torch.int64:
                return IntegerType.get_signless(64, self._ctx)
            case torch.uint8:
                return IntegerType.get_unsigned(8, self._ctx)
            case torch.uint16:
                return IntegerType.get_unsigned(16, self._ctx)
            case torch.uint32:
                return IntegerType.get_unsigned(32, self._ctx)
            case torch.uint64:
                return IntegerType.get_unsigned(64, self._ctx)
            case torch.qint32:
                if not isinstance(dtype, TypeInfo):
                    raise ValueError("TypeInfo required for qint32")
                if dtype.scale is None or dtype.zero_point is None:
                    raise ValueError("scale and zero_point required for qint32")
                return quant.UniformQuantizedType.get(
                    quant.UniformQuantizedType.FLAG_SIGNED,
                    IntegerType.get_signless(32, self._ctx),
                    F32Type.get(self._ctx),
                    dtype.scale,
                    dtype.zero_point,
                    torch.iinfo(torch.qint32).min,
                    torch.iinfo(torch.qint32).max,
                )
            case torch.qint8:
                if not isinstance(dtype, TypeInfo):
                    raise ValueError("TypeInfo required for qint8")
                if dtype.scale is None or dtype.zero_point is None:
                    raise ValueError("scale and zero_point required for qint8")
                return quant.UniformQuantizedType.get(
                    quant.UniformQuantizedType.FLAG_SIGNED,
                    IntegerType.get_signless(8, self._ctx),
                    F32Type.get(self._ctx),
                    dtype.scale,
                    dtype.zero_point,
                    torch.iinfo(torch.qint8).min,
                    torch.iinfo(torch.qint8).max,
                )
            case torch.quint8:
                if not isinstance(dtype, TypeInfo):
                    raise ValueError("TypeInfo required for quint8")
                if dtype.scale is None or dtype.zero_point is None:
                    raise ValueError("scale and zero_point required for quint8")
                return quant.UniformQuantizedType.get(
                    0,
                    IntegerType.get_unsigned(8, self._ctx),
                    F32Type.get(self._ctx),
                    dtype.scale,
                    dtype.zero_point,
                    torch.iinfo(torch.quint8).min,
                    torch.iinfo(torch.quint8).max,
                )
            case torch.bool:
                return IntegerType.get_signless(1, self._ctx)
            case _:
                raise TypeError(f"Invalid Type {dtype}")

    def _get_torch_dtype_from_type(self, mlir_type: Type) -> torch.dtype:
        """Convert MLIR Type to torch.dtype.
        Parameters
        ----------
        mlir_type : Type
            MLIR type to convert
        Returns
        -------
        torch.dtype
            Corresponding torch dtype
        """
        type_str = str(mlir_type)

        if isinstance(mlir_type, BF16Type) or type_str == "bf16":
            return torch.bfloat16
        elif isinstance(mlir_type, F16Type) or type_str == "f16":
            return torch.float16
        elif isinstance(mlir_type, F32Type) or type_str == "f32":
            return torch.float32
        elif isinstance(mlir_type, F64Type) or type_str == "f64":
            return torch.float64
        elif isinstance(mlir_type, IntegerType):
            width = mlir_type.width
            is_signed = mlir_type.is_signed
            is_unsigned = mlir_type.is_unsigned

            if width == 1:
                return torch.bool
            elif width == 8:
                if is_unsigned:
                    return torch.uint8
                else:
                    return torch.int8
            elif width == 16:
                if is_unsigned:
                    return torch.uint16
                else:
                    return torch.int16
            elif width == 32:
                if is_unsigned:
                    return torch.uint32
                else:
                    return torch.int32
            elif width == 64:
                if is_unsigned:
                    return torch.uint64
                else:
                    return torch.int64
            else:
                raise TypeError(f"Unsupported integer width: {width}")
        else:
            raise TypeError(f"Unsupported MLIR type: {mlir_type}")

    def _get_next_global_id(self) -> int:
        """Return a monotonically increasing integer used to disambiguate op locations."""
        self._global_id += 1
        return self._global_id

    def _get_loc_of_extra_file_callee(self, id: int = 0) -> Location:
        """Create a ``Location`` referencing the first caller outside this file.

        Walks the call stack until it finds a frame whose filename differs from
        the immediate caller, then encodes ``filename:lineno:id(N)`` as an
        MLIR named location.  This provides human-readable provenance for ops
        generated through the builder.

        Parameters
        ----------
        id : int
            A numeric identifier appended to the location string (typically the
            global op counter from :meth:`_get_next_global_id`).

        Returns
        -------
        Location
            An MLIR ``Location`` encoding the external call-site.

        Raises
        ------
        RuntimeError
            If the entire call stack resides in the same file.
        """
        stack = inspect.stack()
        caller_filename = stack[1].filename

        while len(stack) > 0 and stack[0].filename == caller_filename:
            stack = stack[1:]

        if len(stack) == 0:
            raise RuntimeError(
                "Top of callstack to builder funcs must be outside the caller's file"
            )

        return Location.name(
            f"{stack[0].filename}:{str(stack[0].lineno)}:id({str(id)})"
        )

    def _get_loc_from_str(self, loc: Union[str, Location]) -> Location:
        """Normalise *loc* to an MLIR ``Location``, wrapping plain strings."""
        if isinstance(loc, str):
            return Location.name(loc)
        else:
            return loc

    def _create_ranked_tensor_type(
        self,
        shape: Shape,
        data_type: Optional[Union[Type, torch.dtype]] = None,
        encoding: Optional[Attribute] = None,
    ) -> RankedTensorType:
        """Build a ``RankedTensorType`` from a shape, element type, and optional encoding.

        If *data_type* is a ``torch.dtype`` it is converted to the corresponding
        MLIR type first.  If ``None``, defaults to ``f32``.

        Parameters
        ----------
        shape : Shape
            Tensor dimensions.
        data_type : Union[Type, torch.dtype], optional
            Element type.  Defaults to ``F32Type``.
        encoding : Attribute, optional
            Layout encoding attribute (e.g. ``TTNNLayoutAttr``).

        Returns
        -------
        RankedTensorType
        """
        with self._ctx, self._loc:
            if isinstance(data_type, torch.dtype):
                dtype = self._get_type_from_torch_dtype(data_type)
            else:
                dtype = data_type if data_type is not None else F32Type.get(self._ctx)
            return RankedTensorType.get(shape, dtype, encoding)

    def _organize_eltwise_golden(self, inputs: List[Operand]) -> List[GoldenMapTensor]:
        """Default golden-argument organiser for elementwise ops.

        Returns the golden tensors in the same order as the input operands.
        Subclasses may override for ops that need different argument layouts.
        """
        return [self._goldens[inp] for inp in inputs]

    def _generate_random_tensor(
        self, shape: Shape, dtype: Union[torch.dtype, TypeInfo]
    ) -> torch.Tensor:
        """Generate a random ``torch.Tensor`` of the given *shape* and *dtype*.

        Floating-point tensors use a standard-normal distribution, boolean
        tensors are uniform in ``{0, 1}``, integer tensors span the full range
        of their type, and quantised types are generated from a normal
        distribution before quantisation.
        """
        if isinstance(dtype, TypeInfo):
            float_tensor = torch.randn(shape, dtype=torch.float32)
            return torch.quantize_per_tensor(
                float_tensor, dtype.scale, dtype.zero_point, dtype.dtype
            )
        if dtype.is_floating_point:
            return torch.randn(shape, dtype=dtype)
        elif dtype == torch.bool:
            return torch.randint(0, 2, shape, dtype=torch.bool)
        else:
            min_int = torch.iinfo(dtype).min
            max_int = torch.iinfo(dtype).max
            return torch.randint(
                low=min_int,
                high=max_int,
                size=shape,
                dtype=dtype,
            )

    def _generate_golden_tensor(
        self, operand: Operand, dtype: Union[torch.dtype, TypeInfo]
    ) -> GoldenMapTensor:
        """Create a random :class:`GoldenMapTensor` matching the shape of *operand*."""
        random_tensor = self._generate_random_tensor(self.get_shape(operand), dtype)
        return GoldenMapTensor({0: random_tensor}, mesh_shape=self._mesh_shape)

    def _generate_golden_device_tensor(
        self, loc: str, golden_map_tensor: GoldenMapTensor
    ) -> Dict[int, GoldenTensor]:
        """Convert a :class:`GoldenMapTensor` into per-device :class:`GoldenTensor` objects.

        Each device shard is converted to the C++ ``GoldenTensor`` handle
        expected by the runtime, keyed by device ID.

        Parameters
        ----------
        loc : str
            Location string used to tag the golden tensor for runtime lookup.
        golden_map_tensor : GoldenMapTensor
            The high-level golden tensor to convert.

        Returns
        -------
        Dict[int, GoldenTensor]
            Mapping from device ID to its ``GoldenTensor``.
        """
        device_golden_info: Dict[int, GoldenTensor] = {}
        contiguous_tensor = golden_map_tensor.contiguous()
        for device_id, device_golden in contiguous_tensor.shard_map.items():
            data_type = self._get_datatype_from_torch_dtype(device_golden.dtype)
            device_golden_info[device_id] = GoldenTensor(
                loc,
                list(device_golden.shape),
                list(device_golden.stride()),
                data_type if data_type is not None else DataType.Float32,
                device_golden.data_ptr(),
                device_golden.numel() * device_golden.dtype.itemsize,
            )

        return device_golden_info

    def _create_builder_golden_from_torch_tensor(
        self,
        inputs: Dict[Operand, Union[Callable, torch.Tensor, Dict[int, torch.Tensor]]],
    ) -> Dict[Operand, GoldenMapTensor]:
        """Wrap user-provided tensors (or callables / shard-maps) into :class:`GoldenMapTensor`.

        Supports three input forms per operand:

        * **Callable** – called with the operand's shape, must return a
          ``torch.Tensor``.
        * **torch.Tensor** – assigned to device 0.
        * **Dict[int, torch.Tensor]** – explicit per-device shard map.

        Returns
        -------
        Dict[Operand, GoldenMapTensor]
            Normalised golden tensors ready for the builder's internal storage.
        """
        input_goldens: Dict[Operand, GoldenMapTensor] = {}
        for operand, tensor_or_shard_map_or_callable in inputs.items():
            if callable(tensor_or_shard_map_or_callable):
                operand_shape = self.get_shape(operand)
                try:
                    generated_tensor = tensor_or_shard_map_or_callable(operand_shape)
                    if not isinstance(generated_tensor, torch.Tensor):
                        raise TypeError(
                            f"Callable must return a torch.Tensor, got {type(generated_tensor)}"
                        )
                except Exception as e:
                    raise RuntimeError(
                        f"Error calling initialization function for operand {operand}: {e}"
                    )
                golden_tensor = GoldenMapTensor(
                    {0: generated_tensor}, mesh_shape=self._mesh_shape
                )
            elif isinstance(tensor_or_shard_map_or_callable, torch.Tensor):
                golden_tensor = GoldenMapTensor(
                    {0: tensor_or_shard_map_or_callable}, mesh_shape=self._mesh_shape
                )
            else:
                golden_tensor = GoldenMapTensor(
                    tensor_or_shard_map_or_callable, mesh_shape=self._mesh_shape
                )
            input_goldens[operand] = golden_tensor

        return input_goldens

    def _set_golden_tensor(
        self,
        operand: Operand,
        goldens: List[GoldenMapTensor],
    ):
        """Store a golden tensor for *operand* and record its location string."""
        self._goldens[operand] = goldens
        self._operand_to_loc[operand] = str(operand.location)

    def _set_goldens(
        self,
        goldens: Dict[Operand, GoldenMapTensor],
    ):
        """Batch-register multiple golden tensors at once."""
        for operand, golden in goldens.items():
            self._set_golden_tensor(operand, golden)

    def _get_golden_tensor(
        self,
        operand: Operand,
    ) -> GoldenMapTensor:
        """Retrieve the golden tensor previously stored for *operand*."""
        return self._goldens[operand]

    def _get_golden_tensors(
        self,
        operands: List[Operand],
    ) -> List[GoldenMapTensor]:
        """Retrieve golden tensors for a list of operands, preserving order."""
        return [self._goldens[operand] for operand in operands]

    def _get_location(self) -> Location:
        """Create a ``Location`` from the grandparent call-site (two frames up).

        This is used by public builder methods so that the MLIR location
        points at the test or user code that invoked the builder, rather than
        at the builder implementation itself.
        """
        stack = inspect.stack()
        caller_frame = stack[2]
        filename = caller_frame.filename
        lineno = caller_frame.lineno
        return Location.name(f"{filename}:{lineno}")

    # ----- Shared Empty Operations -----

    def _empty(self, shape: Shape, data_type: Optional[Type] = None) -> OpView:
        """Create an empty operation using the dialect-specific EmptyOp."""
        dtype = data_type if data_type is not None else F32Type.get(self._ctx)
        return self._create_empty_from_tensor_type(
            shape, self._create_ranked_tensor_type(shape, dtype)
        )

    def _create_empty_from_tensor_type(
        self, shape: Shape, tensor_type: RankedTensorType
    ) -> OpView:
        """Create empty operation from tensor type using dialect-specific EmptyOp."""
        with self._ctx, self._loc:
            op = self._get_empty_op(tensor_type)
            return op

    def _get_empty_op(self, tensor_type: RankedTensorType) -> OpView:
        """Get dialect-specific empty operation. Must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _get_empty_op")

    def create_tensor_encoding(
        self, shape: Shape, element_type: Union[torch.dtype, TypeInfo]
    ) -> ttnn.ir.TTNNLayoutAttr:
        """Create a layout encoding attribute for the given tensor shape and element type.

        The base implementation raises ``NotImplementedError``; subclasses
        (e.g. ``TTNNBuilder``) override this to return the appropriate encoding.

        Parameters
        ----------
        shape : Shape
            Tensor dimensions.
        element_type : Union[torch.dtype, TypeInfo]
            Element dtype.

        Returns
        -------
        ttnn.ir.TTNNLayoutAttr
            Layout encoding, or ``None`` for dialects that do not use one.
        """
        raise NotImplementedError("Subclasses must implement create_tensor_encoding")

    # ----- Shared Metal Tensor Layout -----

    def get_metal_tensor_layout(
        self,
        logical_shape: Shape,
        tiled=False,
        oobVal=None,  # Will default to ttcore.OOBVal.Undef in the utility
        memorySpace=None,  # Will default to ttcore.MemorySpace.DeviceL1 in the utility
        grid: Optional[Tuple[int, int]] = None,
        index_map: Optional[AffineMap] = None,
        memory_layout=None,  # Will default to ttcore.TensorMemoryLayout.Sharded in the utility
        dim_alignments: Optional[Tuple[int, ...]] = None,
    ):
        """Create a metal tensor layout using the shared implementation."""
        from builder.base.builder_apis import get_metal_tensor_layout
        from ttmlir.dialects import ttcore

        # Set defaults if not provided
        if oobVal is None:
            oobVal = ttcore.OOBVal.Undef
        if memorySpace is None:
            memorySpace = ttcore.MemorySpace.DeviceL1
        if memory_layout is None:
            memory_layout = ttcore.TensorMemoryLayout.Sharded

        return get_metal_tensor_layout(
            self._ctx,
            logical_shape,
            tiled,
            oobVal,
            memorySpace,
            grid,
            index_map,
            memory_layout,
            dim_alignments,
        )

    # ----- Operations -----

    def call(
        self,
        nested_func: Callable,
        original_inputs: List[Operand],
        loc: Optional[str] = None,
    ):
        """Emit a ``func.CallOp`` that invokes *nested_func* as a private function.

        *nested_func* is captured into a new ``func.FuncOp`` (with ``private``
        visibility) and inserted at the current module insertion point.  A
        ``func.CallOp`` is then emitted at the current position, forwarding
        *original_inputs*.  Golden tensors from the caller are propagated into
        the nested function and back out again.

        Parameters
        ----------
        nested_func : Callable
            A Python function whose signature matches ``(*inputs, builder) -> result``.
        original_inputs : List[Operand]
            Operands to pass as arguments to the nested function.
        loc : str, optional
            Custom MLIR location string.  If ``None``, the call-site location
            is inferred from the Python call stack.

        Returns
        -------
        Union[OpResult, Tuple[OpResult, ...]]
            The result(s) of the ``func.CallOp``.
        """
        fn_input_types = []
        for operand in original_inputs:
            fn_input_types.append(operand.type)

        ordered_inputs = []
        ordered_outputs = []

        with InsertionPoint(self._current_module_insertion_point):

            @func.func(*fn_input_types, name=nested_func.__name__)
            def decorated_func(*inputs):
                input_goldens: Dict[Operand, GoldenMapTensor] = {}
                for index, (new_operand, original_operand) in enumerate(
                    zip(inputs, original_inputs)
                ):
                    input_goldens[new_operand] = self._get_golden_tensor(
                        original_operand
                    )

                self._set_goldens(input_goldens)
                ordered_inputs.extend(inputs)

                result = nested_func(*inputs, self)

                outputs = result if hasattr(result, "__iter__") else [result]
                output_goldens: Dict[Operand, GoldenMapTensor] = {}
                for op in outputs:
                    output_goldens[op] = self._get_golden_tensor(op)
                self._set_goldens(output_goldens)
                ordered_outputs.extend(outputs)

                return process_multi_return_result(result)

        new_func_op = decorated_func.func_op
        new_func_op.sym_visibility = StringAttr.get("private")
        self._func_ops_generated[new_func_op] = [ordered_inputs, ordered_outputs]

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        call_op = func.CallOp(new_func_op, original_inputs, loc=loc)
        for index, output in enumerate(ordered_outputs):
            self._set_golden_tensor(
                call_op.results[index], self._get_golden_tensor(output)
            )

        return (
            call_op.results[0] if len(call_op.results) == 1 else tuple(call_op.results)
        )

    # ----- Parse module ----

    def _build_op_from_parsed_op(
        self,
        parsed_op: Operation,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Dispatch a parsed op to its registered ``@parse`` handler.

        This is the main entry point of the round-trip parser: it looks up the
        correct parser method via :meth:`get_parser_from_opview` and delegates
        to it, passing the old→new operand mapping in *global_dict*.

        Parameters
        ----------
        parsed_op : Operation
            An operation from the source module being re-emitted.
        global_dict : Dict[Operand, Operand]
            Mapping from operands in the parsed module to their counterparts
            in the new module being constructed.

        Returns
        -------
        Tuple[Operation, Dict[OpResult, OpResult]]
            The newly created operation and a mapping from old results to new
            results (used to update *global_dict*).
        """
        if isinstance(parsed_op, func.CallOp):
            return self.parse_call_op(parsed_op, global_dict)

        parsed_function = self.get_parser_from_opview(type(parsed_op))
        return parsed_function(self, parsed_op, global_dict)

    def get_input_types(self, func_op: func.FuncOp):
        """Extract ``RankedTensorType`` values for every argument of *func_op*.

        The returned list preserves the argument order and reconstructs each
        type with its shape, element type, and encoding.

        Parameters
        ----------
        func_op : func.FuncOp
            The function operation to inspect.

        Returns
        -------
        List[RankedTensorType]
            One tensor type per function argument.

        Raises
        ------
        ValueError
            If any argument type is not a ``RankedTensorType``.
        """
        inputs_types = []
        inputs_shapes = []
        input_encodings = []
        for arg in func_op.type.inputs:
            if isinstance(arg, RankedTensorType):
                inputs_types.append(arg.element_type)
                inputs_shapes.append(arg.shape)
                input_encodings.append(arg.encoding)
            else:
                raise ValueError("Only ranked tensor types are supported")

        return [
            self._create_ranked_tensor_type(shape, dtype, encoding)
            for (shape, dtype, encoding) in zip(
                inputs_shapes, inputs_types, input_encodings
            )
        ]

    def parse_root_module(
        self, parsed_root_module: Module, golden_inputs: Dict[str, [List[torch.tensor]]]
    ):
        """Re-emit an entire parsed MLIR module through the builder, attaching goldens.

        This is the top-level entry point for the "load → rebuild → verify"
        round-trip workflow.  It walks *parsed_root_module* looking for
        ``ttcore.CPUModuleOp``, ``ttcore.DeviceModuleOp``, and top-level
        ``func.FuncOp`` entries, re-emitting each through the builder's
        registered parsers while propagating golden tensors from
        *golden_inputs*.

        Parameters
        ----------
        parsed_root_module : Module
            The MLIR module to re-emit (typically loaded from a ``.mlir`` file).
        golden_inputs : Dict[str, List[torch.Tensor]]
            Mapping from function name to a list of input tensors used to seed
            golden values during the rebuild.

        Returns
        -------
        Module
            A freshly constructed MLIR module with golden tensors attached.
        """
        found_cpu_module = False

        for entry in parsed_root_module.body.operations:
            if isinstance(entry, ttcore.CPUModuleOp):
                found_cpu_module = True
                builtin_module = entry.regions[0].blocks[0].operations[0]
                for op in builtin_module.regions[0].blocks[0].operations:
                    if isinstance(op, func.FuncOp):
                        self._hoisted_cpu_functions.append(op.name.value)
                        self._func_name_to_op[op.name.value] = op
            elif isinstance(entry, ttcore.DeviceModuleOp):
                builtin_module = entry.regions[0].blocks[0].operations[0]
                for op in builtin_module.regions[0].blocks[0].operations:
                    if isinstance(op, func.FuncOp):
                        # Only add functions with bodies, not declarations
                        if not op.is_external:
                            self._func_name_to_op[op.name.value] = op

                        for block in op.body:
                            for inner_op in block.operations:
                                if isinstance(inner_op, func.CallOp):
                                    self._nested_funcs.append(inner_op.callee.value)
            elif isinstance(entry, func.FuncOp):
                self._func_name_to_op[entry.name.value] = entry

                for block in entry.body:
                    for inner_op in block.operations:
                        if isinstance(inner_op, func.CallOp):
                            self._nested_funcs.append(inner_op.callee.value)

        new_root_module = Module.create()
        self._root_module_insertion_point = new_root_module.body
        self._current_module_insertion_point = new_root_module.body

        with InsertionPoint(new_root_module.body):
            if found_cpu_module:
                cpu_module_op = ttcore.CPUModuleOp()
                region = cpu_module_op.regions[0]
                block = Block.create_at_start(region)
                new_module = Module.create()
                cloned_op = new_module.operation.clone()
                cpu_module_op.regions[0].blocks[0].append(cloned_op.operation)
                self._cpu_module_insertion_point = cloned_op.regions[0].blocks[0]

            for entry in parsed_root_module.body.operations:
                if isinstance(entry, ttcore.DeviceModuleOp):
                    device_module_op = ttcore.DeviceModuleOp()
                    region = device_module_op.regions[0]
                    block = Block.create_at_start(region)
                    new_builtin_module = self.parse_builtin_module(
                        entry.regions[0].blocks[0].operations[0], golden_inputs
                    )
                    device_module_op.regions[0].blocks[0].append(
                        new_builtin_module.operation
                    )
                elif isinstance(entry, func.FuncOp):
                    if entry.name.value in self._nested_funcs:
                        continue
                    self.parse_func(entry, golden_inputs)

        return new_root_module

    def parse_builtin_module(
        self,
        parsed_builtin_module: Module,
        golden_inputs: Dict[str, [List[torch.tensor]]],
    ):
        """Re-emit the inner ``builtin.module`` within a ``DeviceModuleOp``.

        Parameters
        ----------
        parsed_builtin_module : Module
            The nested module to re-emit.
        golden_inputs : Dict[str, List[torch.Tensor]]
            Golden input tensors, keyed by function name.

        Returns
        -------
        Operation
            A cloned ``builtin.module`` operation with rebuilt function bodies.
        """
        new_builtin_module = Module.create()
        cloned_op = new_builtin_module.operation.clone()
        self._current_module_insertion_point = cloned_op.regions[0].blocks[0]

        with InsertionPoint(cloned_op.regions[0].blocks[0]):
            for entry in parsed_builtin_module.regions[0].blocks[0].operations:
                if isinstance(entry, func.FuncOp):
                    if entry.name.value in self._nested_funcs:
                        continue
                    new_func = self.parse_func(entry, golden_inputs)

        return cloned_op

    def parse_func(
        self, parsed_func: func.FuncOp, golden_inputs: Dict[str, [List[torch.tensor]]]
    ):
        """Re-emit a single ``func.FuncOp`` through the builder, replaying every op.

        If *golden_inputs* contains an entry for this function's name, those
        tensors are used as input goldens.  Otherwise random tensors matching
        each argument's shape and dtype are generated.

        Parameters
        ----------
        parsed_func : func.FuncOp
            The function to re-emit.
        golden_inputs : Dict[str, List[torch.Tensor]]
            Optional pre-supplied golden tensors keyed by function name.

        Returns
        -------
        func.FuncOp
            The newly created function operation.
        """
        fn_input_types = self.get_input_types(parsed_func)

        parsed_func_golden_inputs = []
        if parsed_func.name.value in golden_inputs.keys():
            parsed_func_golden_inputs = golden_inputs[parsed_func.name.value]
        else:
            for ttype in fn_input_types:
                shape = ttype.shape
                dtype = self._get_torch_dtype_from_type(ttype.element_type)

                if dtype.is_floating_point or dtype.is_complex:
                    if len(shape) == 0:
                        golden_input = torch.randn(1, dtype=dtype).squeeze()
                    else:
                        golden_input = torch.randn(*shape, dtype=dtype)
                    parsed_func_golden_inputs.append(golden_input)
                elif dtype == torch.bool:
                    if len(shape) == 0:
                        golden_input = torch.randint(0, 2, (), dtype=dtype)
                    else:
                        golden_input = torch.randint(0, 2, shape, dtype=dtype)
                    parsed_func_golden_inputs.append(golden_input)
                else:
                    if len(shape) == 0:
                        golden_input = torch.randint(0, 256, (), dtype=dtype)
                    else:
                        golden_input = torch.randint(0, 256, shape, dtype=dtype)
                    parsed_func_golden_inputs.append(golden_input)

        ordered_inputs = []
        ordered_outputs = []

        @func.func(*fn_input_types, name=parsed_func.name.value)
        def decorated_func(*inputs):
            golden_dict = {}
            for operand, torch_golden in zip(inputs, parsed_func_golden_inputs):
                golden_dict[operand] = torch_golden

            input_goldens: Dict[
                Operand, GoldenMapTensor
            ] = self._create_builder_golden_from_torch_tensor(golden_dict)
            self._set_goldens(input_goldens)
            ordered_inputs.extend(inputs)

            global_dict = {}
            for i, arg in enumerate(parsed_func.arguments):
                global_dict[arg] = inputs[i]

            global_result = None
            for block in parsed_func.body:
                for op in block.operations:
                    if isinstance(op, func.ReturnOp):
                        global_result = tuple(
                            global_dict[operand] for operand in op.operands
                        )
                    elif isinstance(op, ttir.EmptyOp):
                        continue
                    else:
                        (
                            parsed_op,
                            op_golden_dictionary,
                        ) = self._build_op_from_parsed_op(op, global_dict)
                        global_dict.update(op_golden_dictionary)

            outputs = (
                global_result
                if hasattr(global_result, "__iter__")
                else (global_result,)
            )
            output_goldens: Dict[Operand, GoldenMapTensor] = {}
            for op in outputs:
                output_goldens[op] = self._get_golden_tensor(op)
            self._set_goldens(output_goldens)
            ordered_outputs.extend(outputs)

            return process_multi_return_result(global_result)

        new_func_op = decorated_func.func_op
        self._func_ops_generated[new_func_op] = [ordered_inputs, ordered_outputs]
        return new_func_op

    def parse_nested_func(
        self, parsed_func: func.FuncOp, golden_inputs: List[GoldenMapTensor]
    ):
        """Re-emit a nested (called) function with pre-computed golden inputs.

        Unlike :meth:`parse_func`, golden inputs are provided directly as
        :class:`GoldenMapTensor` objects (propagated from the caller's
        ``CallOp`` handler).  The resulting ``func.FuncOp`` is tagged with
        ``tt.function_type = "forward_cpu"`` for CPU-hosted functions.

        Parameters
        ----------
        parsed_func : func.FuncOp
            The function to re-emit.
        golden_inputs : List[GoldenMapTensor]
            One golden tensor per function argument, in order.

        Returns
        -------
        func.FuncOp
            The newly created function operation.
        """
        fn_input_types = self.get_input_types(parsed_func)

        ordered_inputs = []
        ordered_outputs = []

        @func.func(*fn_input_types, name=parsed_func.name.value)
        def decorated_func(*inputs):
            input_goldens = {}
            for operand, golden_map_tensor in zip(inputs, golden_inputs):
                input_goldens[operand] = golden_map_tensor

            self._set_goldens(input_goldens)
            ordered_inputs.extend(inputs)

            global_dict = {}
            for i, arg in enumerate(parsed_func.arguments):
                global_dict[arg] = inputs[i]

            global_result = None
            for block in parsed_func.body:
                for op in block.operations:
                    if isinstance(op, func.ReturnOp):
                        global_result = tuple(
                            global_dict[operand] for operand in op.operands
                        )
                    else:
                        (
                            parsed_op,
                            op_golden_dictionary,
                        ) = self._build_op_from_parsed_op(op, global_dict)
                        global_dict.update(op_golden_dictionary)

            outputs = (
                global_result
                if hasattr(global_result, "__iter__")
                else (global_result,)
            )
            output_goldens: Dict[Operand, GoldenMapTensor] = {}
            for op in outputs:
                output_goldens[op] = self._get_golden_tensor(op)
            self._set_goldens(output_goldens)
            ordered_outputs.extend(outputs)

            return process_multi_return_result(global_result)

        new_func_op = decorated_func.func_op
        new_func_op.attributes["tt.function_type"] = StringAttr.get(
            "forward_cpu", self._ctx
        )
        self._func_ops_generated[new_func_op] = [ordered_inputs, ordered_outputs]
        return new_func_op

    def parse_call_op(
        self,
        parsed_op: func.CallOp,
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[Operand, GoldenMapTensor]]:
        """Re-emit a ``func.CallOp``, rebuilding the callee when necessary.

        Handles two cases:

        * **Hoisted CPU calls** – the callee is re-emitted into the CPU module
          and a ``private`` declaration is placed in the device module.  The
          ``ttir.cpu_hoisted_call`` unit attribute is preserved on the new
          ``CallOp``.
        * **Regular nested calls** – the callee is re-emitted at the current
          module insertion point.

        Parameters
        ----------
        parsed_op : func.CallOp
            The original ``CallOp`` from the parsed module.
        global_dict : Dict[Operand, Operand]
            Old-to-new operand mapping.

        Returns
        -------
        Tuple[Operation, Dict[Operand, GoldenMapTensor]]
            The new ``CallOp`` and a result-mapping dictionary.
        """
        is_hoisted = False
        parsed_op_attributes = parsed_op.attributes
        parsed_op_callee_value = parsed_op.callee.value
        parsed_op_operands = parsed_op.operands

        for attr in parsed_op_attributes:
            if attr.name == "ttir.cpu_hoisted_call":
                is_hoisted = True
                break

        if is_hoisted:
            insertion_point = self._cpu_module_insertion_point
            hoisted_func_name = parsed_op_callee_value
            nested_func_op = self._func_name_to_op[hoisted_func_name]

            new_golden_inputs = []
            for operand in parsed_op_operands:
                owner0 = operand.owner
                if isinstance(owner0, Block):
                    queried_operand = operand
                else:
                    queried_operand = owner0.result

                new_golden_inputs.append(
                    self._get_golden_tensor(global_dict[queried_operand])
                )

            with InsertionPoint(insertion_point):
                new_func_op = self.parse_nested_func(nested_func_op, new_golden_inputs)

            with InsertionPoint(self._current_module_insertion_point):
                private_func_op = func.FuncOp(
                    type=new_func_op.type, name=hoisted_func_name, visibility="private"
                )
                private_func_op.attributes["tt.function_type"] = StringAttr.get(
                    "forward_cpu_declaration", self._ctx
                )

            self._nested_funcs.append(private_func_op.name.value)

            new_operands = [global_dict[operand] for operand in parsed_op_operands]
            call_op = func.CallOp(private_func_op, new_operands)
            call_op_result = call_op.results
            call_op.attributes["ttir.cpu_hoisted_call"] = UnitAttr.get(self._ctx)

            ordered_inputs, ordered_outputs = self._func_ops_generated[new_func_op]
            for index, output in enumerate(ordered_outputs):
                self._set_golden_tensor(
                    call_op_result[index], self._get_golden_tensor(output)
                )

            op_map_dictionary = {}
            parsed_op_results = parsed_op.results
            for old_result, new_result in zip(parsed_op_results, call_op.results):
                op_map_dictionary[old_result] = new_result

            return call_op, op_map_dictionary

        else:
            insertion_point = self._current_module_insertion_point
            nested_func_op = self._func_name_to_op[parsed_op.callee.value]
            new_golden_inputs = []
            for operand in parsed_op.operands:
                owner0 = operand.owner
                if isinstance(owner0, Block):
                    queried_operand = operand
                else:
                    queried_operand = owner0.result

                new_golden_inputs.append(
                    self._get_golden_tensor(global_dict[queried_operand])
                )

            with InsertionPoint(insertion_point):
                new_func_op = self.parse_nested_func(nested_func_op, new_golden_inputs)

            new_operands = [global_dict[operand] for operand in parsed_op.operands]
            call_op = func.CallOp(new_func_op, new_operands)

            ordered_inputs, ordered_outputs = self._func_ops_generated[new_func_op]
            for index, output in enumerate(ordered_outputs):
                self._set_golden_tensor(
                    call_op.results[index], self._get_golden_tensor(output)
                )

            op_map_dictionary = {}
            for old_result, new_result in zip(parsed_op.results, call_op.results):
                op_map_dictionary[old_result] = new_result

        return call_op, op_map_dictionary

    def split_call_op(
        self,
        old_op: func.CallOp,
    ) -> Tuple[Module, TTIRBuilder]:
        """Recursively split a ``CallOp`` into individual per-op modules.

        For non-hoisted calls, this walks the callee's body and delegates each
        inner operation to its registered ``@split`` handler, producing a list
        of ``(Module, Builder)`` pairs.

        Parameters
        ----------
        old_op : func.CallOp
            The call operation to split.

        Returns
        -------
        List[Tuple[Module, TTIRBuilder]]
            One ``(module, builder)`` pair per inner operation.
        """
        is_hoisted = False
        for attr in old_op.attributes:
            if attr.name == "ttir.cpu_hoisted_call":
                is_hoisted = True
                break

        if is_hoisted:
            sub_modules_and_builders = []
        else:
            nested_func_op = self._func_name_to_op[old_op.callee.value]
            sub_modules_and_builders = []

            for block in nested_func_op.body:
                for op in block.operations:
                    if isinstance(op, func.ReturnOp) or isinstance(
                        op,
                        ttir.EmptyOp,
                    ):
                        continue
                    elif isinstance(op, func.CallOp):
                        sub_op_module_builder = self.split_call_op(op)
                        sub_modules_and_builders.append(sub_op_module_builder)
                    else:
                        sub_op_module_builder = self.split_op(op)
                        sub_modules_and_builders.append(sub_op_module_builder)

        return sub_modules_and_builders

    # ----- Helper decorator functions ----

    def func(self, input_shapes: List[List[int]], input_types: List[torch.dtype]):
        """Decorator that wraps a Python function into a ``func.FuncOp``.

        The decorated function receives MLIR block arguments (one per shape)
        and a builder reference.  It should build ops and return one or more
        results.  Golden tensors are auto-generated from the declared shapes
        and dtypes unless golden checking is disabled.

        Parameters
        ----------
        input_shapes : List[List[int]]
            Shape of each input tensor.
        input_types : List[torch.dtype]
            Element dtype of each input tensor.

        Returns
        -------
        Callable
            A decorator that captures the function body as MLIR IR.

        Example
        -------
        ::

            @builder.func([[64, 64]], [torch.float32])
            def my_op(in0, builder):
                return builder.relu(in0)
        """
        def wrapper(fn):
            encoding_fn = self.create_tensor_encoding
            fn_input_types = [
                self._create_ranked_tensor_type(
                    shape,
                    self._get_type_from_torch_dtype(dtype),
                    encoding_fn(shape, dtype) if encoding_fn else None,
                )
                for shape, dtype in zip(input_shapes, input_types)
            ]

            ordered_inputs = []
            ordered_outputs = []

            @func.func(*fn_input_types, name=fn.__name__)
            def decorated_func(*inputs):
                if not self._disable_golden_check:
                    input_goldens: Dict[Operand, GoldenMapTensor] = {}
                    for index, (operand, dtype) in enumerate(zip(inputs, input_types)):
                        input_goldens[operand] = self._generate_golden_tensor(
                            operand, dtype
                        )
                    self._set_goldens(input_goldens)
                ordered_inputs.extend(inputs)

                result = fn(*inputs, self)

                outputs = result if hasattr(result, "__iter__") else [result]

                if not self._disable_golden_check:
                    output_goldens: Dict[Operand, GoldenMapTensor] = {}
                    for op in outputs:
                        output_goldens[op] = self._get_golden_tensor(op)
                    self._set_goldens(output_goldens)
                ordered_outputs.extend(outputs)

                return process_multi_return_result(result)

            new_func_op = decorated_func.func_op
            self._func_ops_generated[new_func_op] = [ordered_inputs, ordered_outputs]
            return new_func_op

        return wrapper

    def device_module(self, root_func: Callable):
        """Wrap *root_func*'s output inside a ``ttcore.DeviceModuleOp``.

        Creates the device-module scaffolding (region, block, nested
        ``builtin.module``) and calls *root_func(self)* within the inner
        insertion point.

        Parameters
        ----------
        root_func : Callable
            A function ``(builder) -> func.FuncOp`` that emits ops into the
            device module.

        Returns
        -------
        ttcore.DeviceModuleOp
        """
        def wrapper(self):
            device_module_op = ttcore.DeviceModuleOp()
            region = device_module_op.regions[0]
            block = Block.create_at_start(region)
            new_module = Module.create()
            cloned_op = new_module.operation.clone()
            self._current_module_insertion_point = cloned_op.regions[0].blocks[0]

            with InsertionPoint(cloned_op.regions[0].blocks[0]):
                new_func = root_func(self)

            device_module_op.regions[0].blocks[0].append(cloned_op.operation)
            return device_module_op

        return wrapper(self)

    def cpu_module(self, root_func: Callable):
        """Wrap *root_func*'s output inside a ``ttcore.CPUModuleOp``.

        Analogous to :meth:`device_module` but targets the CPU-hosted module
        used for functions that should run on the host rather than the device.

        Parameters
        ----------
        root_func : Callable
            A function ``(builder) -> func.FuncOp`` that emits ops into the
            CPU module.

        Returns
        -------
        ttcore.CPUModuleOp
        """
        def wrapper(self):
            cpu_module_op = ttcore.CPUModuleOp()
            region = cpu_module_op.regions[0]
            block = Block.create_at_start(region)
            new_module = Module.create()
            cloned_op = new_module.operation.clone()
            self._current_module_insertion_point = cloned_op.regions[0].blocks[0]

            with InsertionPoint(cloned_op.regions[0].blocks[0]):
                new_func = root_func(self)

            cpu_module_op.regions[0].blocks[0].append(cloned_op.operation)
            return cpu_module_op

        return wrapper(self)

    # ----- Debug dialect operations ----

    @tag(debug.AnnotateOp)
    def annotate(
        self,
        operand: Operand,
        annotation: str,
        loc: Optional[str] = None,
    ) -> OpResult:
        """Create a ``debug.AnnotateOp`` that attaches a human-readable *annotation* to *operand*.

        The annotation is a pass-through: the output tensor is identical to the
        input but carries the annotation string as metadata.

        Parameters
        ----------
        operand : Operand
            Input tensor to annotate.
        annotation : str
            Free-form annotation text.
        loc : str, optional
            Custom MLIR location string.

        Returns
        -------
        OpResult
            The annotated tensor (same data as *operand*).
        """
        debug_op = self.get_opview_from_method(Builder.annotate)
        annotation_attr = StringAttr.get(annotation)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = debug_op(
            operand,
            annotation=annotation_attr,
            loc=loc,
        )
        op_result = op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(debug_op)
            golden_output = op_golden_function(input0, annotation_attr)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @tag(debug.BreakpointOp)
    def breakpoint(
        self,
        operand: Operand,
        loc: Optional[str] = None,
    ) -> OpResult:
        """Create a ``debug.BreakpointOp`` that signals a runtime debug breakpoint.

        Parameters
        ----------
        operand : Operand
            Input tensor (passed through unchanged).
        loc : str, optional
            Custom MLIR location string.

        Returns
        -------
        OpResult
            The same tensor as *operand*.
        """
        debug_op = self.get_opview_from_method(Builder.breakpoint)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = debug_op(
            operand,
            loc=loc,
        )
        op_result = op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(debug_op)
            golden_output = op_golden_function(input0)
            self._set_golden_tensor(op_result, golden_output)

        return op_result

    @tag(debug.MemorySnapshotOp)
    def memory_snapshot(
        self,
        operand: Operand,
        file_path: str,
        loc: Optional[str] = None,
    ) -> OpResult:
        """Create a ``debug.MemorySnapshotOp`` that dumps device memory to a file.

        At runtime the device memory state at this point in the graph is
        serialised to *file_path* for offline analysis.

        Parameters
        ----------
        operand : Operand
            Input tensor (passed through unchanged).
        file_path : str
            Destination path for the memory snapshot.
        loc : str, optional
            Custom MLIR location string.

        Returns
        -------
        OpResult
            The same tensor as *operand*.
        """
        debug_op = self.get_opview_from_method(Builder.memory_snapshot)
        file_path_attr = StringAttr.get(file_path)

        if loc is None:
            loc = self._get_location()
        else:
            loc = Location.name(loc)

        op = debug_op(
            operand,
            file_path=file_path_attr,
            loc=loc,
        )
        op_result = op.result

        if not self._disable_golden_check:
            input0 = self._get_golden_tensor(operand)
            op_golden_function = get_golden_function(debug_op)
            golden_output = op_golden_function(input0, file_path_attr)
            self._set_golden_tensor(op_result, golden_output)

        return op_result
