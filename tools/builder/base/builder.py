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
    """
    Metaclass for Builder that automatically registers operation builders.

    This metaclass is responsible for building lookup maps that connect MLIR
    operation views (OpView) to their corresponding builder, parser, and split
    methods. When a Builder subclass is created, the metaclass scans all methods
    for special decorators (@tag, @parse, @split) and populates the appropriate
    dictionaries.

    The maps enable dynamic dispatch from MLIR operations to their handler
    methods, which is essential for both constructing new operations and
    parsing existing MLIR modules.
    """

    def __new__(mcls, name, bases, namespace):
        """
        Create a new Builder class and register its operation handlers.

        Parameters
        ----------
        mcls : type
            The metaclass (BuilderMeta).
        name : str
            Name of the class being created.
        bases : tuple
            Base classes of the new class.
        namespace : dict
            Class namespace containing methods and attributes.

        Returns
        -------
        type
            The newly created Builder subclass with populated operation maps.
        """
        cls = super().__new__(mcls, name, bases, namespace)
        cls.build_opview_to_builder_map()
        cls.build_opview_to_parser_map()
        cls.build_opview_to_split(map)
        return cls


class Builder(metaclass=BuilderMeta):
    """
    Base class for constructing MLIR modules with golden tensor verification.

    The Builder class provides infrastructure for programmatically constructing
    MLIR operations while simultaneously computing and tracking golden (reference)
    tensors for verification. It manages the mapping between MLIR operands and
    their corresponding PyTorch tensors, enabling end-to-end correctness checking.

    This class serves as the foundation for dialect-specific builders (e.g.,
    TTIRBuilder, TTNNBuilder) that implement the actual operation construction
    methods.

    Attributes
    ----------
    opview_to_builder_map : Dict[OpView, Callable]
        Maps MLIR operation views to their builder methods.
    opview_to_parser_map : Dict[OpView, Callable]
        Maps MLIR operation views to their parser methods.
    opview_to_split_map : Dict[OpView, Callable]
        Maps MLIR operation views to their split methods.

    Examples
    --------
    Subclasses should implement dialect-specific operations::

        class TTIRBuilder(Builder):
            @tag(ttir.AddOp)
            def add(self, lhs: Operand, rhs: Operand) -> OpResult:
                # Implementation here
                pass
    """

    opview_to_builder_map: Dict[OpView, Callable] = {}
    opview_to_parser_map: Dict[OpView, Callable] = {}
    opview_to_split_map: Dict[OpView, Callable] = {}

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
        """
        Initialize a new Builder instance.

        Parameters
        ----------
        ctx : Context
            MLIR context for creating operations and types.
        location : Location
            Default MLIR location for operations created by this builder.
        mesh_name : Union[List[str], str], optional
            Name(s) of the device mesh(es). Defaults to "mesh".
        mesh_dict : Union[List[OrderedDict[str, int]], OrderedDict[str, int]], optional
            Dictionary specifying mesh dimensions. Keys are dimension names
            (e.g., "x", "y") and values are sizes. Defaults to a 1x1 mesh.
        disable_golden_check : bool, optional
            If True, skip golden tensor computation and verification.
            Defaults to False.

        Raises
        ------
        ValueError
            If mesh_name and mesh_dict have different lengths when both are lists.
        """
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
        """
        Scan class methods and register those decorated with @tag.

        This method populates the opview_to_builder_map dictionary by finding
        all methods that have been decorated with the @tag decorator, which
        associates them with specific MLIR operation views.
        """
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            func = attr

            if callable(attr) and hasattr(func, "_tag"):
                cls.opview_to_builder_map[func._tag] = attr

    @classmethod
    def build_opview_to_parser_map(cls):
        """
        Scan class methods and register those decorated with @parse.

        This method populates the opview_to_parser_map dictionary by finding
        all methods that have been decorated with the @parse decorator, which
        associates them with specific MLIR operation views for parsing.
        """
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            func = attr

            if callable(attr) and hasattr(func, "_parse"):
                cls.opview_to_parser_map[func._parse] = attr

    @classmethod
    def build_opview_to_split(cls, map):
        """
        Scan class methods and register those decorated with @split.

        This method populates the opview_to_split_map dictionary by finding
        all methods that have been decorated with the @split decorator, which
        associates them with specific MLIR operation views for splitting.
        """
        for attr_name in dir(cls):
            attr = getattr(cls, attr_name)
            func = attr

            if callable(attr) and hasattr(func, "_split"):
                cls.opview_to_split_map[func._split] = attr

    def get_opview_from_method(self, method: func) -> OpView:
        """
        Extract the OpView type from a method decorated with @tag.

        Parameters
        ----------
        method : Callable
            A method that was decorated with @tag.

        Returns
        -------
        OpView or None
            The MLIR operation view type associated with the method,
            or None if the method was not decorated.
        """
        return getattr(method, "_tag", None)

    def get_opview_from_parser(self, parser: func) -> OpView:
        """
        Extract the OpView type from a method decorated with @parse.

        Parameters
        ----------
        parser : Callable
            A method that was decorated with @parse.

        Returns
        -------
        OpView or None
            The MLIR operation view type associated with the parser,
            or None if the method was not decorated.
        """
        return getattr(parser, "_parse", None)

    def get_opview_from_split(self, split: func) -> OpView:
        """
        Extract the OpView type from a method decorated with @split.

        Parameters
        ----------
        split : Callable
            A method that was decorated with @split.

        Returns
        -------
        OpView or None
            The MLIR operation view type associated with the split method,
            or None if the method was not decorated.
        """
        return getattr(split, "_split", None)

    def get_builder_from_opview(self, opview: OpView) -> Callable:
        """
        Look up the builder method for a given MLIR operation view.

        Parameters
        ----------
        opview : OpView
            The MLIR operation view type to look up.

        Returns
        -------
        Callable
            The builder method that can construct this operation type.

        Raises
        ------
        AssertionError
            If no builder is registered for the given opview.
        """
        if opview not in self.opview_to_builder_map:
            assert False, f"No builder found for opview {opview}"
        return self.opview_to_builder_map.get(opview)

    def get_parser_from_opview(self, opview: OpView) -> Callable:
        """
        Look up the parser method for a given MLIR operation view.

        Parameters
        ----------
        opview : OpView
            The MLIR operation view type to look up.

        Returns
        -------
        Callable
            The parser method that can reconstruct this operation type.

        Raises
        ------
        AssertionError
            If no parser is registered for the given opview.
        """
        if opview not in self.opview_to_parser_map:
            assert False, f"No parser found for opview {opview}"
        return self.opview_to_parser_map.get(opview)

    def get_split_from_opview(self, opview: OpView) -> Callable:
        """
        Look up the split method for a given MLIR operation view.

        Parameters
        ----------
        opview : OpView
            The MLIR operation view type to look up.

        Returns
        -------
        Callable
            The split method for this operation type.

        Raises
        ------
        AssertionError
            If no split function is registered for the given opview.
        """
        if opview not in self.opview_to_split_map:
            assert False, f"No split function found for opview {opview}"
        return self.opview_to_split_map.get(opview)

    # ----- Public methods -----

    @property
    def context(self) -> Context:
        """
        Get the MLIR context associated with this builder.

        Returns
        -------
        Context
            The MLIR context used for creating operations and types.
        """
        return self._ctx

    @property
    def location(self) -> Location:
        """
        Get the default MLIR location for this builder.

        Returns
        -------
        Location
            The default location used for operations created by this builder.
        """
        return self._loc

    @property
    def mesh_shape(self) -> Tuple[int, int]:
        """
        Get the device mesh shape.

        Returns
        -------
        Tuple[int, int]
            The shape of the device mesh as (x, y) dimensions.
        """
        return self._mesh_shape

    @property
    def golden_map(
        self,
    ) -> Tuple[
        Dict[int, Dict[str, Dict[int, GoldenMapTensor]]],
        Dict[str, Dict[int, GoldenMapTensor]],
    ]:
        """
        Get the golden tensor map for verification.

        This property compiles and returns the golden (reference) tensors
        that have been computed during module construction. The returned
        maps can be used to verify correctness of the compiled program
        against expected outputs.

        Returns
        -------
        Tuple[Dict, Dict]
            A tuple containing:
            - input_output_golden_info: Nested dict mapping program index to
              location to device_id to GoldenMapTensor for inputs/outputs.
            - intermediate_golden_info: Dict mapping location string to
              device_id to GoldenMapTensor for intermediate values.

        Notes
        -----
        If golden checking is disabled or no goldens are marked for storage,
        empty dictionaries are returned.
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
        """
        Get the shape of an operand's tensor type.

        Parameters
        ----------
        input : Operand
            The MLIR operand (BlockArgument or OpResult) to query.

        Returns
        -------
        Shape
            The shape of the operand's tensor type as a list or tuple of ints.
        """
        return self._get_type(input).shape

    def get_type(self, input: Operand) -> Type:
        """
        Get the element type of an operand's tensor type.

        Parameters
        ----------
        input : Operand
            The MLIR operand (BlockArgument or OpResult) to query.

        Returns
        -------
        Type
            The MLIR element type (e.g., F32Type, BF16Type) of the tensor.
        """
        return self._get_type(input).element_type

    def set_goldens(
        self,
        inputs: Dict[Operand, Union[Callable, torch.tensor, Dict[int : torch.tensor]]],
        outputs: Dict[Operand, Union[torch.tensor, Dict[int : torch.tensor]]] = None,
        set_all_outputs: bool = True,
    ):
        """
        Set golden (reference) tensors for input and output operands.

        This method associates PyTorch tensors with MLIR operands for later
        verification. The tensors can be provided directly, as callables that
        generate tensors, or as device-sharded dictionaries.

        Parameters
        ----------
        inputs : Dict[Operand, Union[Callable, torch.tensor, Dict[int, torch.tensor]]]
            Mapping from input operands to their golden tensors. Values can be:
            - torch.Tensor: Used directly
            - Callable: Called with operand shape to generate tensor
            - Dict[int, torch.Tensor]: Device ID to tensor mapping for sharded data
        outputs : Dict[Operand, Union[torch.tensor, Dict[int, torch.tensor]]], optional
            Mapping from output operands to their expected golden tensors.
        set_all_outputs : bool, optional
            If True, mark all provided outputs for verification. Defaults to True.
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
        """
        Set golden tensors using pre-constructed GoldenMapTensor objects.

        This is an alternative to set_goldens() when you already have
        GoldenMapTensor instances rather than raw PyTorch tensors.

        Parameters
        ----------
        inputs : Dict[Operand, GoldenMapTensor]
            Mapping from input operands to their GoldenMapTensor objects.
        outputs : Dict[Operand, GoldenMapTensor], optional
            Mapping from output operands to their expected GoldenMapTensor objects.
        """
        self._set_goldens(inputs)

        if outputs != None:
            self.set_goldens_to_check(outputs.keys())
            self._set_goldens(outputs)

    def set_operand_goldens(
        self, operands: Dict[Operand, Union[torch.tensor, Dict[int : torch.tensor]]]
    ):
        """
        Set golden tensors for operands and mark them for verification.

        This is a convenience method that combines setting goldens and marking
        them for verification in a single call.

        Parameters
        ----------
        operands : Dict[Operand, Union[torch.tensor, Dict[int, torch.tensor]]]
            Mapping from operands to their golden tensors or device shard maps.
        """
        self._set_goldens(self._create_builder_golden_from_torch_tensor(operands))
        self.set_goldens_to_check(operands.keys())

    def set_goldens_to_check(self, operands: List[Operand], override: bool = False):
        """
        Mark operands for golden verification.

        Parameters
        ----------
        operands : List[Operand]
            List of operands whose golden values should be verified.
        override : bool, optional
            If True, replace the existing list. If False, append to it.
            Defaults to False.
        """
        if override:
            self._goldens_to_store = operands
        else:
            self._goldens_to_store.extend(operands)

    def set_graph_level_check(self, check: bool):
        """
        Enable or disable graph-level golden checking.

        When enabled, only input/output goldens are verified, skipping
        intermediate value verification.

        Parameters
        ----------
        check : bool
            True to enable graph-level checking only, False for full checking.
        """
        self._force_graph_level_check = check

    def bypass(self, operand: Operand):
        """
        Mark an operation to be bypassed during golden comparison.

        This is useful for operations whose outputs cannot be meaningfully
        compared (e.g., random number generation).

        Parameters
        ----------
        operand : Operand
            The operand whose defining operation should be bypassed.

        Raises
        ------
        TypeError
            If the operand is a BlockArgument (function inputs cannot be bypassed).
        """
        if isinstance(operand, BlockArgument):
            raise TypeError("Cannot bypass BlockArgument")

        loc = str(operand.owner.location)
        self._bypass_ops.append(loc)

    def set_arg_attribute(
        self, operand: Operand, new_attr_name: str, new_attr: Attribute
    ):
        """
        Add or update an attribute on a function argument.

        Parameters
        ----------
        operand : Operand
            A BlockArgument representing a function parameter.
        new_attr_name : str
            Name of the attribute to set.
        new_attr : Attribute
            The MLIR attribute value to assign.

        Notes
        -----
        This modifies the arg_attrs of the containing FuncOp in place.
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
        """
        Compute the output shape and type by running the golden function.

        This method executes the golden (reference) implementation of an
        operation to determine what shape and dtype the output should have.
        This is necessary because TTIR operations do not have MLIR shape
        inference traits.

        Parameters
        ----------
        organize_golden_args : Callable
            Function to transform input operands into golden function arguments.
        inputs : List[Operand]
            List of input operands to the operation.
        op_function : Callable
            The MLIR operation function being built.
        golden_kwargs : dict, optional
            Additional keyword arguments to pass to the golden function.

        Returns
        -------
        Tuple[Shape, torch.dtype] or None
            The output shape and dtype, or None if no golden function exists.
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
        """
        Convert a PyTorch dtype to a tt-mlir DataType enum value.

        Parameters
        ----------
        dtype : torch.dtype
            The PyTorch data type to convert.

        Returns
        -------
        DataType
            The corresponding tt-mlir DataType enum value.
        """
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
        """
        Get the MLIR type of an operand.

        Parameters
        ----------
        input : Operand
            The operand to query.

        Returns
        -------
        RankedTensorType
            The MLIR RankedTensorType of the operand.
        """
        return input.type

    def _get_type_from_torch_dtype(
        self,
        dtype: Union[torch.dtype, TypeInfo],
        scale: Optional[float] = None,
        zero_point: Optional[float] = None,
    ) -> Type:
        """
        Convert a PyTorch dtype to an MLIR Type.

        Supports standard floating-point, integer, and quantized types.
        For quantized types, a TypeInfo object with scale and zero_point
        must be provided.

        Parameters
        ----------
        dtype : Union[torch.dtype, TypeInfo]
            The PyTorch dtype or TypeInfo for quantized types.
        scale : float, optional
            Quantization scale (used with dtype if TypeInfo not provided).
        zero_point : float, optional
            Quantization zero point (used with dtype if TypeInfo not provided).

        Returns
        -------
        Type
            The corresponding MLIR type.

        Raises
        ------
        ValueError
            If quantized type is requested without proper TypeInfo.
        TypeError
            If the dtype is not supported.
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
        """
        Generate the next unique global identifier.

        Returns
        -------
        int
            A monotonically increasing integer ID.
        """
        self._global_id += 1
        return self._global_id

    def _get_loc_of_extra_file_callee(self, id: int = 0) -> Location:
        """
        Create an MLIR Location from the external call site.

        This method walks up the call stack to find the first frame that
        is outside the current file, and creates a location string from
        that frame's filename and line number.

        Parameters
        ----------
        id : int, optional
            An additional identifier to include in the location string.

        Returns
        -------
        Location
            An MLIR named location with format "filename:lineno:id(N)".

        Raises
        ------
        RuntimeError
            If the entire call stack is within the same file.
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
        """
        Convert a string or Location to an MLIR Location.

        Parameters
        ----------
        loc : Union[str, Location]
            Either a string to convert or an existing Location.

        Returns
        -------
        Location
            An MLIR Location object.
        """
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
        """
        Create an MLIR RankedTensorType.

        Parameters
        ----------
        shape : Shape
            The shape of the tensor.
        data_type : Union[Type, torch.dtype], optional
            The element type. Can be an MLIR Type or torch.dtype.
            Defaults to F32Type.
        encoding : Attribute, optional
            Optional tensor encoding attribute (e.g., for layout info).

        Returns
        -------
        RankedTensorType
            The constructed MLIR tensor type.
        """
        with self._ctx, self._loc:
            if isinstance(data_type, torch.dtype):
                dtype = self._get_type_from_torch_dtype(data_type)
            else:
                dtype = data_type if data_type is not None else F32Type.get(self._ctx)
            return RankedTensorType.get(shape, dtype, encoding)

    def _organize_eltwise_golden(self, inputs: List[Operand]) -> List[GoldenMapTensor]:
        """
        Retrieve golden tensors for element-wise operation inputs.

        Parameters
        ----------
        inputs : List[Operand]
            List of input operands.

        Returns
        -------
        List[GoldenMapTensor]
            List of corresponding golden tensors.
        """
        return [self._goldens[inp] for inp in inputs]

    def _generate_random_tensor(
        self, shape: Shape, dtype: Union[torch.dtype, TypeInfo]
    ) -> torch.Tensor:
        """
        Generate a random PyTorch tensor with the specified shape and dtype.

        For floating-point types, generates values from a standard normal
        distribution. For integer types, generates values across the full
        range. For quantized types, generates and quantizes random floats.

        Parameters
        ----------
        shape : Shape
            The shape of the tensor to generate.
        dtype : Union[torch.dtype, TypeInfo]
            The data type, or TypeInfo for quantized types.

        Returns
        -------
        torch.Tensor
            A randomly initialized tensor.
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
        """
        Generate a random golden tensor for an operand.

        Creates a random tensor matching the operand's shape and wraps it
        in a GoldenMapTensor for multi-device verification.

        Parameters
        ----------
        operand : Operand
            The MLIR operand to generate a golden tensor for.
        dtype : Union[torch.dtype, TypeInfo]
            Data type for the generated tensor.

        Returns
        -------
        GoldenMapTensor
            A golden tensor wrapper containing the random tensor.
        """
        random_tensor = self._generate_random_tensor(self.get_shape(operand), dtype)
        return GoldenMapTensor({0: random_tensor}, mesh_shape=self._mesh_shape)

    def _generate_golden_device_tensor(
        self, loc: str, golden_map_tensor: GoldenMapTensor
    ) -> Dict[int, GoldenTensor]:
        """
        Convert a GoldenMapTensor to device-specific GoldenTensor objects.

        Transforms the high-level golden tensor representation into the
        low-level format required by the runtime for verification.

        Parameters
        ----------
        loc : str
            Location string identifying where this tensor is used.
        golden_map_tensor : GoldenMapTensor
            The golden tensor to convert.

        Returns
        -------
        Dict[int, GoldenTensor]
            Mapping from device ID to GoldenTensor for runtime verification.
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
        """
        Convert various tensor input formats to GoldenMapTensor objects.

        Handles three input formats:
        - Callable: Called with operand shape to generate a tensor
        - torch.Tensor: Used directly as golden tensor
        - Dict[int, torch.Tensor]: Device-sharded tensor map

        Parameters
        ----------
        inputs : Dict[Operand, Union[Callable, torch.Tensor, Dict[int, torch.Tensor]]]
            Mapping from operands to their golden tensor specifications.

        Returns
        -------
        Dict[Operand, GoldenMapTensor]
            Mapping from operands to GoldenMapTensor wrappers.

        Raises
        ------
        TypeError
            If a callable returns a non-tensor value.
        RuntimeError
            If an error occurs calling the initialization function.
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
        """
        Store a golden tensor for an operand.

        Parameters
        ----------
        operand : Operand
            The MLIR operand to associate with the golden tensor.
        goldens : List[GoldenMapTensor]
            The golden tensor(s) for verification.
        """
        self._goldens[operand] = goldens
        self._operand_to_loc[operand] = str(operand.location)

    def _set_goldens(
        self,
        goldens: Dict[Operand, GoldenMapTensor],
    ):
        """
        Store multiple golden tensors at once.

        Parameters
        ----------
        goldens : Dict[Operand, GoldenMapTensor]
            Mapping from operands to their golden tensors.
        """
        for operand, golden in goldens.items():
            self._set_golden_tensor(operand, golden)

    def _get_golden_tensor(
        self,
        operand: Operand,
    ) -> GoldenMapTensor:
        """
        Retrieve the golden tensor for an operand.

        Parameters
        ----------
        operand : Operand
            The operand to look up.

        Returns
        -------
        GoldenMapTensor
            The stored golden tensor for the operand.
        """
        return self._goldens[operand]

    def _get_golden_tensors(
        self,
        operands: List[Operand],
    ) -> List[GoldenMapTensor]:
        """
        Retrieve golden tensors for multiple operands.

        Parameters
        ----------
        operands : List[Operand]
            List of operands to look up.

        Returns
        -------
        List[GoldenMapTensor]
            List of golden tensors in the same order as the operands.
        """
        return [self._goldens[operand] for operand in operands]

    def _get_location(self) -> Location:
        """
        Get an MLIR Location from the caller's call site.

        Returns
        -------
        Location
            An MLIR named location with format "filename:lineno".
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
        """
        Create a tensor encoding attribute for the given shape and element type.

        This is an abstract method that must be implemented by subclasses to
        create dialect-specific tensor layout attributes. The encoding defines
        how tensor data is laid out in memory.

        Parameters
        ----------
        shape : Shape
            The logical shape of the tensor.
        element_type : Union[torch.dtype, TypeInfo]
            The element type of the tensor, either as a PyTorch dtype or
            a TypeInfo object containing dtype and optional quantization info.

        Returns
        -------
        ttnn.ir.TTNNLayoutAttr
            The tensor layout attribute for the dialect.

        Raises
        ------
        NotImplementedError
            Always raised; subclasses must provide an implementation.
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
        """
        Create a metal tensor layout using the shared implementation.

        This method creates a MetalLayoutAttr that specifies how tensor data
        is organized in device memory, including sharding, tiling, and memory
        space configuration.

        Parameters
        ----------
        logical_shape : Shape
            The logical shape of the tensor.
        tiled : bool, optional
            Whether to use tiled layout with 32x32 tiles. Default is False.
        oobVal : ttcore.OOBVal, optional
            Out-of-bounds value handling. Default is ttcore.OOBVal.Undef.
        memorySpace : ttcore.MemorySpace, optional
            Memory space for the tensor. Default is ttcore.MemorySpace.DeviceL1.
        grid : Tuple[int, int], optional
            Grid shape for sharding. Default is None (uses 1x1 grid).
        index_map : AffineMap, optional
            Affine map for layout transformation. Default is None.
        memory_layout : ttcore.TensorMemoryLayout, optional
            Tensor memory layout type. Default is ttcore.TensorMemoryLayout.Sharded.
        dim_alignments : Tuple[int, ...], optional
            Explicit dimension alignments for padding. Default is None.

        Returns
        -------
        RankedTensorType
            The metal tensor type with the specified layout configuration.
        """
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
        """
        Create a function call operation with golden tensor propagation.

        This method creates a new MLIR function from the provided Python
        callable, wraps it in a FuncOp, and generates a CallOp to invoke it.
        Golden tensors are automatically propagated from the original inputs
        through the nested function to its outputs.

        Parameters
        ----------
        nested_func : Callable
            A Python function that takes operands and a builder, and returns
            one or more output operands. The function will be converted to
            an MLIR FuncOp.
        original_inputs : List[Operand]
            The input operands to pass to the function call.
        loc : str, optional
            Location string for the call operation. If None, auto-generated.

        Returns
        -------
        Union[OpResult, Tuple[OpResult, ...]]
            The result(s) of the call operation. Returns a single OpResult
            if the function has one output, or a tuple for multiple outputs.
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
        if isinstance(parsed_op, func.CallOp):
            return self.parse_call_op(parsed_op, global_dict)

        parsed_function = self.get_parser_from_opview(type(parsed_op))
        return parsed_function(self, parsed_op, global_dict)

    def get_input_types(self, func_op: func.FuncOp):
        """
        Extract input types from a FuncOp and create corresponding tensor types.

        This method inspects the function signature to extract the shapes,
        element types, and encodings of all input arguments, then creates
        new RankedTensorType instances for each input.

        Parameters
        ----------
        func_op : func.FuncOp
            The MLIR function operation to extract input types from.

        Returns
        -------
        List[RankedTensorType]
            List of ranked tensor types corresponding to each function input.

        Raises
        ------
        ValueError
            If any input argument is not a RankedTensorType.
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
        """
        Parse and rebuild a root MLIR module with golden tensor propagation.

        This method traverses the structure of an existing MLIR module,
        identifies CPU and device modules, extracts function operations,
        and rebuilds the module structure while propagating golden tensors
        through all operations.

        Parameters
        ----------
        parsed_root_module : Module
            The existing MLIR module to parse and rebuild.
        golden_inputs : Dict[str, List[torch.Tensor]]
            Dictionary mapping function names to lists of golden input tensors.
            If a function is not in this dictionary, random inputs are generated.

        Returns
        -------
        Module
            A new MLIR module with the same structure but with golden tensor
            tracking enabled for all operations.
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
        """
        Parse and rebuild a builtin module within a device or CPU module.

        This method processes all function operations within a builtin module,
        skipping nested functions that will be processed separately, and
        rebuilds each function with golden tensor propagation.

        Parameters
        ----------
        parsed_builtin_module : Module
            The builtin module to parse, typically found inside a DeviceModuleOp
            or CPUModuleOp.
        golden_inputs : Dict[str, List[torch.Tensor]]
            Dictionary mapping function names to lists of golden input tensors.

        Returns
        -------
        Operation
            The cloned and rebuilt builtin module operation.
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
        """
        Parse and rebuild a function operation with golden tensor tracking.

        This method creates a new FuncOp that mirrors the structure of the
        parsed function. It sets up golden tensors for inputs (either from
        the provided dictionary or randomly generated), processes each
        operation in the function body, and tracks golden tensors through
        all intermediate results.

        Parameters
        ----------
        parsed_func : func.FuncOp
            The function operation to parse and rebuild.
        golden_inputs : Dict[str, List[torch.Tensor]]
            Dictionary mapping function names to lists of golden input tensors.
            If the function name is not found, random tensors are generated
            based on the function signature.

        Returns
        -------
        func.FuncOp
            The newly created function operation with golden tensor tracking.
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
        """
        Parse and rebuild a nested function with pre-computed golden inputs.

        This method is similar to parse_func but is designed for nested
        functions called via CallOp. It takes pre-computed GoldenMapTensor
        objects instead of raw torch tensors, allowing golden values to
        propagate from the caller into the nested function.

        Parameters
        ----------
        parsed_func : func.FuncOp
            The nested function operation to parse and rebuild.
        golden_inputs : List[GoldenMapTensor]
            List of GoldenMapTensor objects corresponding to each function
            input, propagated from the calling context.

        Returns
        -------
        func.FuncOp
            The newly created function operation with golden tensor tracking
            and the tt.function_type attribute set to forward_cpu.
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
        """
        Parse and rebuild a function call operation with golden propagation.

        This method handles both regular nested function calls and hoisted
        CPU function calls. For hoisted calls (marked with ttir.cpu_hoisted_call
        attribute), the callee is processed into the CPU module. For regular
        calls, the callee is processed inline.

        Parameters
        ----------
        parsed_op : func.CallOp
            The call operation to parse.
        global_dict : Dict[Operand, Operand]
            Dictionary mapping operands from the original module to their
            corresponding operands in the rebuilt module.

        Returns
        -------
        Tuple[Operation, Dict[Operand, GoldenMapTensor]]
            A tuple containing the new call operation and a dictionary mapping
            the original result operands to their new counterparts.
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
        """
        Split a call operation into separate sub-modules for each nested op.

        This method recursively traverses a nested function call and splits
        each operation (including nested calls) into separate module/builder
        pairs. This is used for operation-level isolation and testing.

        Parameters
        ----------
        old_op : func.CallOp
            The call operation to split.

        Returns
        -------
        List[Tuple[Module, Builder]]
            List of tuples, where each tuple contains a module and builder
            for a single operation from the nested function. Returns an
            empty list for hoisted CPU calls.
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
        """
        Decorator for creating MLIR functions with automatic golden tensor generation.

        This decorator transforms a Python function into an MLIR FuncOp. Input
        tensors are automatically created with random golden values, and the
        decorated function can use builder operations to construct the function
        body.

        Parameters
        ----------
        input_shapes : List[List[int]]
            List of shapes for each input tensor argument.
        input_types : List[torch.dtype]
            List of PyTorch dtypes for each input tensor argument.

        Returns
        -------
        Callable
            A decorator that wraps a function and returns its FuncOp.

        Examples
        --------
        ::

            @builder.func([[64, 128], [64, 128]], [torch.float32, torch.float32])
            def my_add(in0: Operand, in1: Operand, builder: TTIRBuilder):
                return builder.add(in0, in1)
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
        """
        Create a device module containing the given function.

        This method wraps a function in a DeviceModuleOp, which represents
        code that will execute on Tenstorrent accelerator devices.

        Parameters
        ----------
        root_func : Callable
            A function that takes a builder and creates MLIR operations.

        Returns
        -------
        ttcore.DeviceModuleOp
            The device module operation containing the generated code.
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
        """
        Create a CPU module containing the given function.

        This method wraps a function in a CPUModuleOp, which represents
        code that will execute on the host CPU rather than accelerators.

        Parameters
        ----------
        root_func : Callable
            A function that takes a builder and creates MLIR operations.

        Returns
        -------
        ttcore.CPUModuleOp
            The CPU module operation containing the generated code.
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
        """
        Add an annotation to an operand for debugging purposes.

        The annotation is attached as metadata and passed through to the
        output without modifying the tensor values.

        Parameters
        ----------
        operand : Operand
            The tensor operand to annotate.
        annotation : str
            The annotation string to attach.
        loc : str, optional
            Location string for the operation.

        Returns
        -------
        OpResult
            The annotated tensor (same values as input).
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
        """
        Insert a breakpoint operation for debugging.

        This operation can be used to pause execution during debugging
        and inspect tensor values. The tensor values pass through unchanged.

        Parameters
        ----------
        operand : Operand
            The tensor operand to inspect at the breakpoint.
        loc : str, optional
            Location string for the operation.

        Returns
        -------
        OpResult
            The tensor (same values as input).
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
        """
        Capture a memory snapshot to a file for debugging.

        This operation dumps the tensor contents to a file at the specified
        path during execution, useful for post-mortem debugging analysis.

        Parameters
        ----------
        operand : Operand
            The tensor operand to snapshot.
        file_path : str
            Path where the memory snapshot will be written.
        loc : str, optional
            Location string for the operation.

        Returns
        -------
        OpResult
            The tensor (same values as input).
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
