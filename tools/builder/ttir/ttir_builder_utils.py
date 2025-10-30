# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import inspect
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

import pytest

from ttmlir.ir import Context, InsertionPoint, Location, Module, RankedTensorType
from ttmlir.dialects import func

from builder.base.builder import Operand
from builder.ttir.ttir_builder import TTIRBuilder


# ----- Exception Classes -----


class MLIRConversionException(Exception):
    """Exception raised during MLIR to TTIR conversion."""

    pass


class OperationMappingException(Exception):
    """Exception raised when operation mapping fails."""

    pass


class ParameterMappingException(Exception):
    """Exception raised when parameter mapping fails."""

    pass


# ----- Public API -----


def mlir_to_ttir_builder(path_to_file: str) -> Tuple[Module, TTIRBuilder]:
    """
    Load an MLIR file or parse an MLIR string and convert it to TTIR using TTIRBuilder.

    This function dynamically maps MLIR operations to TTIRBuilder methods using
    runtime inspection and generic parameter matching. It handles operand mapping,
    attribute conversion, and special cases for operations with non-standard signatures.

    Parameters
    ----------
    path_to_file : str
        Path to the MLIR file to convert, or MLIR string content directly

    Returns
    -------
    Tuple[Module, TTIRBuilder]
        Tuple containing:
        - Module: The converted TTIR module
        - TTIRBuilder: The builder instance with golden tensors

    Raises
    ------
    MLIRConversionException
        If conversion fails at any stage

    Examples
    --------
    Basic usage with a simple unary operation:

    Given an MLIR file with content:

    .. code-block:: mlir

        module {
          func.func @abs_op(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
            %0 = ttir.empty() : tensor<4x4xf32>
            %1 = ttir.abs %arg0, %0 : tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
            return %1 : tensor<4x4xf32>
          }
        }

    Convert it to a builder:

    >>> module, builder = mlir_to_ttir_builder("abs_example.mlir")
    >>> print(str(module))  # Prints the reconstructed MLIR

    Binary operation with two inputs:

    .. code-block:: mlir

        module {
          func.func @add_op(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
            %0 = ttir.empty() : tensor<4x4xf32>
            %1 = ttir.add %arg0, %arg1, %0 : tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32> -> tensor<4x4xf32>
            return %1 : tensor<4x4xf32>
          }
        }

    >>> module, builder = mlir_to_ttir_builder("add_example.mlir")

    Operation with attributes (transpose):

    .. code-block:: mlir

        module {
          func.func @transpose_op(%arg0: tensor<2x3xf32>) -> tensor<3x2xf32> {
            %0 = ttir.empty() : tensor<3x2xf32>
            %1 = "ttir.transpose"(%arg0, %0) <{dim0 = 0 : i32, dim1 = 1 : i32}> : (tensor<2x3xf32>, tensor<3x2xf32>) -> tensor<3x2xf32>
            return %1 : tensor<3x2xf32>
          }
        }

    >>> module, builder = mlir_to_ttir_builder("transpose_example.mlir")
    >>> # Attributes are automatically extracted and passed to builder methods
    """
    converter = MLIRToTTIRConverter()
    return converter.convert(path_to_file)


# ----- Converter Class -----


class MLIRToTTIRConverter:
    """
    Main converter class for MLIR to TTIR conversion.
    """

    def convert(self, path_to_file: str) -> Tuple[Module, TTIRBuilder]:
        """
        Convert MLIR file to TTIR.

        Parameters
        ----------
        path_to_file : str
            Path to the MLIR file or MLIR string content

        Returns
        -------
        Tuple[Module, TTIRBuilder]
            Converted module and builder instance

        Raises
        ------
        MLIRConversionException
            If conversion fails
        """
        import os

        try:
            ctx = Context()
            with Context() as ctx:
                ctx.allow_unregistered_dialects = True

                # Check if path_to_file is an actual file or MLIR string content
                if os.path.isfile(path_to_file):
                    with open(path_to_file, "r") as mlir_fd:
                        mlir_content = mlir_fd.read()
                else:
                    # Treat it as MLIR string content directly
                    mlir_content = path_to_file

                parsed_module = Module.parse(mlir_content, ctx)
                loc = Location.unknown(ctx)

                with loc:
                    ttir_builder = TTIRBuilder(ctx, loc, (1, 1))
                    new_module = Module.create()

                    target_func = self._find_target_function(parsed_module)
                    input_types = self._extract_input_types(target_func, ttir_builder)

                    with InsertionPoint(new_module.body):

                        @func.func(*input_types, name=target_func.name.value)
                        def converted_func(*inputs):
                            return self._convert_function_body(
                                target_func, inputs, input_types, ttir_builder
                            )

                    return new_module, ttir_builder

        except Exception as e:
            raise MLIRConversionException(f"Failed to convert MLIR: {e}")

    def _find_target_function(self, parsed_module: Module):
        for entry in parsed_module.body.operations:
            if isinstance(entry, func.FuncOp):
                return entry
        raise MLIRConversionException("No function found in MLIR file")

    def _extract_input_types(self, target_func, ttir_builder):
        input_types = []
        for arg_type in target_func.type.inputs:
            if isinstance(arg_type, RankedTensorType):
                input_types.append(
                    ttir_builder._create_ranked_tensor_type(
                        arg_type.shape, arg_type.element_type
                    )
                )
        return input_types

    def _setup_input_goldens(self, inputs, input_types, ttir_builder):
        input_goldens = {}
        for operand, dtype in zip(inputs, input_types):
            tensor_type_str = str(dtype)
            torch_dtype = _mlir_to_torch_dtype(tensor_type_str)
            input_goldens[operand] = ttir_builder._generate_golden_tensor(
                operand, torch_dtype
            )

        ttir_builder._set_goldens(input_goldens)
        ttir_builder._set_input_ordering(inputs)

    def _create_operand_mapping(self, target_func, inputs):
        operand_map = {}
        for i, arg in enumerate(target_func.arguments):
            operand_map[arg] = inputs[i]
        return operand_map

    def _convert_function_body(self, target_func, inputs, input_types, ttir_builder):
        self._setup_input_goldens(inputs, input_types, ttir_builder)
        operand_map = self._create_operand_mapping(target_func, inputs)

        result = None
        for block in target_func.body:
            for op in block.operations:
                result = self._process_operation(op, operand_map, ttir_builder, result)

        self._setup_output_goldens(result, ttir_builder)
        return result

    def _process_operation(self, op, operand_map, ttir_builder, current_result):
        if op.name == "func.return":
            return operand_map.get(op.operands[0]) if op.operands else None
        elif op.name == "ttir.empty":
            op_result = self._convert_operation(op, operand_map, ttir_builder)
            if op.results:
                operand_map[op.results[0]] = op_result
            return current_result

        if op.name.startswith("ttir."):
            op_result = self._convert_operation(op, operand_map, ttir_builder)
            if op.results:
                operand_map[op.results[0]] = op_result
            return current_result
        else:
            raise OperationMappingException(
                f"Operation {op.name} is not a TTIR operation"
            )

    def _convert_operation(self, op, operand_map, ttir_builder):
        try:
            op_name = _get_operation_name_from_op(op)
            method_name = _operation_name_to_method_name(op_name)
            method = getattr(ttir_builder, method_name, None)

            if not method:
                raise OperationMappingException(
                    f"Builder method '{method_name}' not found for operation {op_name}"
                )

            if op_name == "ttir.empty":
                if op.results:
                    result = method(op.results[0].type)
                    result_type_str = str(op.results[0].type)
                    torch_dtype = _mlir_to_torch_dtype(result_type_str)
                    golden = ttir_builder._generate_golden_tensor(result, torch_dtype)
                    ttir_builder._goldens[result] = golden
                    return result
                else:
                    raise OperationMappingException(
                        f"ttir.empty operation has no results"
                    )

            if op_name in ["arange", "zeros", "ones"]:
                operands = []
            else:
                operands = _map_operands_using_builder_method(op, operand_map, method)

            attrs = self._extract_and_map_attributes(op, method)
            self._add_data_type_if_needed(op, method, attrs)
            args, kwargs = self._get_parameter_mapping(
                method_name, method, operands, attrs, op, ttir_builder
            )

            return method(*args, **kwargs)

        except Exception as e:
            raise OperationMappingException(
                f"Failed to convert operation {op.name}: {e}"
            )

    def _get_operands(self, op, operand_map, method):
        operands = []
        for operand in op.operands:
            if operand in operand_map:
                operands.append(operand_map[operand])

        sig = inspect.signature(method)
        required_operand_params = [
            param_name
            for param_name, param in sig.parameters.items()
            if (
                param_name not in ["self", "unit_attrs"]
                and param.kind != inspect.Parameter.KEYWORD_ONLY
                and "Operand" in str(param.annotation)
            )
        ]

        if len(operands) < len(required_operand_params):
            operands = list(operand_map.values())

        return operands

    def _extract_and_map_attributes(self, op, method):
        attrs = _extract_attributes(op)
        sig = inspect.signature(method)
        method_params = set(sig.parameters.keys()) - {"self", "unit_attrs"}
        return _map_attributes(attrs, method_params)

    def _add_data_type_if_needed(self, op, method, attrs):
        sig = inspect.signature(method)
        if "data_type" in sig.parameters and "data_type" not in attrs and op.results:
            result_type_str = str(op.results[0].type)
            attrs["data_type"] = _mlir_to_torch_dtype(result_type_str)

    def _get_parameter_mapping(
        self, method_name, method, operands, attrs, op=None, ttir_builder=None
    ):
        special_handler = _get_special_handler(method_name)
        if special_handler:
            return special_handler(operands, attrs, op, ttir_builder)

        return _match_parameters(method, operands, attrs)

    def _setup_output_goldens(self, result, ttir_builder):
        outputs = result if hasattr(result, "__iter__") else (result,)
        output_goldens = {}
        for op in outputs:
            output_goldens[op] = ttir_builder._get_golden_tensor(op)
        ttir_builder._set_goldens(output_goldens)
        ttir_builder._set_output_ordering(outputs)


# ----- Private Helper Functions -----


def _operation_name_to_method_name(op_name: str) -> str:
    if not op_name.startswith("ttir."):
        raise ValueError(f"Invalid TTIR operation name: {op_name}")

    method_name = op_name[5:]

    special_cases = {
        "isfinite": "is_finite",
        "slice_static": "slice",
        "index_select": "select",
        "empty": "_get_empty_op",
    }

    return special_cases.get(method_name, method_name)


def _get_operation_name_from_op(op) -> str:
    return op.name


def _map_operands_using_builder_method(op, operand_map, method):
    if not method:
        return []

    try:
        sig = inspect.signature(method)
        params = list(sig.parameters.items())

        mapped_operands = []
        operand_idx = 0

        for param_name, param in params:
            if param_name == "self":
                continue

            if param.kind == inspect.Parameter.KEYWORD_ONLY:
                continue

            if param_name in [
                "unit_attrs",
                "broadcast_dimensions",
                "dims",
                "dimension",
                "axis",
                "axes",
            ]:
                continue

            annotation_str = str(param.annotation)
            is_operand_type = any(
                t in annotation_str for t in ["Operand", "OpView", "Value", "IRValue"]
            )

            if is_operand_type and operand_idx < len(op.operands):
                operand = op.operands[operand_idx]

                # Skip last operand if it's from ttir.empty and parameter is optional
                is_last_operand = operand_idx == len(op.operands) - 1
                operand_str = str(operand)
                is_empty_result = "ttir.empty" in operand_str

                if (
                    is_last_operand
                    and is_empty_result
                    and param.default != inspect.Parameter.empty
                ):
                    operand_idx += 1
                    continue

                if operand in operand_map:
                    mapped_operands.append(operand_map[operand])
                operand_idx += 1

        return mapped_operands

    except Exception as e:
        return [
            operand_map[operand] for operand in op.operands if operand in operand_map
        ]


def _mlir_to_torch_dtype(tensor_type_str: str) -> torch.dtype:
    type_mapping = {
        "i32": torch.int32,
        "i64": torch.int64,
        "i16": torch.int16,
        "i8": torch.int8,
        "f64": torch.float64,
        "bf16": torch.bfloat16,
        "f16": torch.float16,
        "i1": torch.bool,
    }

    for mlir_type, torch_type in type_mapping.items():
        if mlir_type in tensor_type_str:
            return torch_type
    return torch.float32


def _extract_attributes(op) -> Dict[str, Any]:
    attrs = {}
    try:
        for named_attr in op.attributes:
            name = str(named_attr.name)
            attr = named_attr.attr

            if hasattr(attr, "value"):
                attrs[name] = attr.value
            elif hasattr(attr, "__getitem__") and hasattr(attr, "__len__"):
                try:
                    attrs[name] = [
                        attr[i].value if hasattr(attr[i], "value") else attr[i]
                        for i in range(len(attr))
                    ]
                except Exception:
                    attrs[name] = list(attr)
            else:
                attrs[name] = attr
    except Exception as e:
        raise MLIRConversionException(
            f"Failed to extract attributes from operation: {e}"
        )

    return attrs


def _map_attributes(attrs: Dict[str, Any], method_params: set) -> Dict[str, Any]:
    mapped_attrs = {}

    for attr_name, attr_value in attrs.items():
        if attr_name in method_params:
            mapped_attrs[attr_name] = attr_value
        else:
            mapped_name = _find_mapped_name(attr_name, method_params)
            mapped_attrs[mapped_name or attr_name] = attr_value

    return mapped_attrs


def _find_mapped_name(attr_name: str, method_params: set) -> Optional[str]:
    if f"{attr_name}_arg" in method_params:
        return f"{attr_name}_arg"

    if attr_name.endswith("_dimensions"):
        base_name = attr_name[:-11]

        if base_name in method_params:
            return base_name

        dimension_param_names = ["dims", "dimensions", "shape", "size"]
        for param_name in dimension_param_names:
            if param_name in method_params:
                return param_name

    return None


def _match_parameters(
    method: Callable, operands: List[Operand], attrs: Dict[str, Any]
) -> Tuple[List, Dict]:
    try:
        sig = inspect.signature(method)
        params = list(sig.parameters.items())

        args = []
        kwargs = {}
        operand_idx = 0

        for param_name, param in params:
            has_default = param.default != inspect.Parameter.empty

            if _is_attribute_only_param(param_name):
                if param_name in attrs:
                    kwargs[param_name] = attrs[param_name]
                continue

            if (
                _is_list_operand_param(param) or _is_list_param_by_name(param_name)
            ) and param.kind != inspect.Parameter.KEYWORD_ONLY:
                args.append(operands)
                operand_idx = len(operands)
            elif (
                operand_idx < len(operands)
                and param.kind != inspect.Parameter.KEYWORD_ONLY
            ):
                args.append(operands[operand_idx])
                operand_idx += 1
            elif param_name in attrs:
                kwargs[param_name] = attrs[param_name]
            elif has_default and operand_idx < len(operands):
                args.append(operands[operand_idx])
                operand_idx += 1

        return args, kwargs

    except Exception as e:
        raise ParameterMappingException(
            f"Failed to match parameters for method {method.__name__}: {e}"
        )


def _is_list_operand_param(param) -> bool:
    if param.annotation == inspect.Parameter.empty:
        return False

    annotation_str = str(param.annotation)

    list_operand_patterns = [
        ("List[" in annotation_str and "Operand" in annotation_str),
        ("list[" in annotation_str and "Operand" in annotation_str),
    ]

    if any(list_operand_patterns):
        return True

    if (
        hasattr(param.annotation, "__origin__")
        and param.annotation.__origin__ is list
        and hasattr(param.annotation, "__args__")
        and len(param.annotation.__args__) > 0
        and "Operand" in str(param.annotation.__args__[0])
    ):
        return True

    return False


def _is_attribute_only_param(param_name: str) -> bool:
    attribute_only_params = ["normalized_shape", "dims", "dimension", "axis", "axes"]
    return param_name in attribute_only_params


def _is_list_param_by_name(param_name: str) -> bool:
    list_param_names = ["ins", "inputs", "operands", "tensors"]
    return param_name in list_param_names


def _get_special_handler(method_name: str) -> Optional[Callable]:
    handlers = {
        "rms_norm": _handle_rms_norm,
        "arange": _handle_arange,
        "zeros": _handle_zeros,
        "ones": _handle_ones,
    }
    return handlers.get(method_name)


def _handle_rms_norm(
    operands: List[Operand], attrs: Dict[str, Any], op=None, ttir_builder=None
) -> Tuple[List, Dict]:
    args = []
    kwargs = {}

    if len(operands) >= 1:
        args = [operands[0]]
    if len(operands) >= 2:
        kwargs["weight"] = operands[1]
    if len(operands) >= 3:
        kwargs["bias"] = operands[2]

    mlir_specific_attrs = ["operandSegmentSizes"]
    for attr_name, attr_value in attrs.items():
        if attr_name not in mlir_specific_attrs:
            kwargs[attr_name] = attr_value

    return args, kwargs


def _handle_arange(
    operands: List[Operand], attrs: Dict[str, Any], op=None, ttir_builder=None
) -> Tuple[List, Dict]:
    args = []
    kwargs = {}

    if op and op.results and ttir_builder:
        result_type = op.results[0].type
        result_operand = ttir_builder._get_empty_op(result_type)

        golden_tensor = ttir_builder._generate_golden_tensor(
            result_operand, torch.float32
        )
        ttir_builder._goldens[result_operand] = golden_tensor

        args = [result_operand]
    elif operands:
        args = [operands[0]]

    mlir_specific_attrs = ["operandSegmentSizes"]
    for attr_name, attr_value in attrs.items():
        if attr_name not in mlir_specific_attrs:
            kwargs[attr_name] = attr_value

    return args, kwargs


def _handle_zeros(
    operands: List[Operand], attrs: Dict[str, Any], op=None, ttir_builder=None
) -> Tuple[List, Dict]:
    args = []
    kwargs = {}

    mlir_specific_attrs = ["operandSegmentSizes"]
    for attr_name, attr_value in attrs.items():
        if attr_name not in mlir_specific_attrs:
            kwargs[attr_name] = attr_value

    return args, kwargs


def _handle_ones(
    operands: List[Operand], attrs: Dict[str, Any], op=None, ttir_builder=None
) -> Tuple[List, Dict]:
    args = []
    kwargs = {}

    mlir_specific_attrs = ["operandSegmentSizes"]
    for attr_name, attr_value in attrs.items():
        if attr_name not in mlir_specific_attrs:
            kwargs[attr_name] = attr_value

    return args, kwargs


# ----- Test Cases -----


@pytest.mark.parametrize(
    "mlir_string",
    [
        # Elementwise unary operations
        """
    module {
        func.func @abs_op(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
            %0 = ttir.empty() : tensor<4x4xf32>
            %1 = "ttir.abs"(%arg0, %0) : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
            return %1 : tensor<4x4xf32>
        }
    }
    """,
        # Elementwise binary operations
        """
    module {
        func.func @multiply_op(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
            %0 = ttir.empty() : tensor<4x4xf32>
            %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
            return %1 : tensor<4x4xf32>
        }
    }
    """,
        # Comparison operations
        """
    module {
        func.func @gt_op(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
            %0 = ttir.empty() : tensor<4x4xf32>
            %1 = "ttir.gt"(%arg0, %arg1, %0) : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
            return %1 : tensor<4x4xf32>
        }
    }
    """,
        # Logical operations
        """
    module {
        func.func @logical_xor_op(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
            %0 = ttir.empty() : tensor<4x4xf32>
            %1 = "ttir.logical_xor"(%arg0, %arg1, %0) : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
            return %1 : tensor<4x4xf32>
        }
    }
    """,
        # Selection operations
        """
    module {
        func.func @where_op(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
            %0 = ttir.empty() : tensor<4x4xf32>
            %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
            return %1 : tensor<4x4xf32>
        }
    }
    """,
        # Bitwise operations
        """
    module {
        func.func @bitwise_and_op(%arg0: tensor<4x4xi32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
            %0 = ttir.empty() : tensor<4x4xi32>
            %1 = "ttir.bitwise_and"(%arg0, %arg1, %0) : (tensor<4x4xi32>, tensor<4x4xi32>, tensor<4x4xi32>) -> tensor<4x4xi32>
            return %1 : tensor<4x4xi32>
        }
    }
    """,
        # Reduction operations
        """
    module {
        func.func @max_op(%arg0: tensor<4x4xf32>) -> tensor<1x1xf32> {
            %0 = ttir.empty() : tensor<1x1xf32>
            %1 = "ttir.max"(%arg0, %0) <{keep_dim = true}> : (tensor<4x4xf32>, tensor<1x1xf32>) -> tensor<1x1xf32>
            return %1 : tensor<1x1xf32>
        }
    }

    """,
        # Tensor manipulation
        """
    module {
        func.func @broadcast_op(%arg0: tensor<1x1x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<1x16x32xf32> {
            %0 = ttir.empty() : tensor<1x16x32xf32>
            %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 1, 16, 1>}> : (tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
            return %1 : tensor<1x16x32xf32>
        }
    }

    """,
        # Neural network operations
        """
    module {
        func.func @matmul_op(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
            %0 = ttir.empty() : tensor<4x4xf32>
            %1 = "ttir.matmul"(%arg0, %arg1, %0) <{transpose_a = false, transpose_b = false}> : (tensor<4x4xf32>, tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
            return %1 : tensor<4x4xf32>
        }
    }
    """,
        # Type operations
        """
    module {
        func.func @typecast_op(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xi32>) -> tensor<4x4xi32> {
            %0 = ttir.empty() : tensor<4x4xi32>
            %1 = "ttir.typecast"(%arg0, %0) <{conservative_folding = false}> : (tensor<4x4xf32>, tensor<4x4xi32>) -> tensor<4x4xi32>
            return %1 : tensor<4x4xi32>
        }
    }
    """,
        # Tensor creation
        """
    module {
        func.func @arange_op(%arg0: tensor<10xf32>) -> tensor<10xf32> {
            %0 = ttir.empty() : tensor<10xf32>
            %1 = "ttir.arange"() <{arange_dimension = 0 : i64, end = 10 : si64, start = 0 : si64, step = 1 : si64}> : () -> tensor<10xf32>
            return %1 : tensor<10xf32>
        }
    }
    """,
        # Quantization operations
        pytest.param(
            """
        module {
            func.func @quantize_op(%arg0: tensor<4x4xf32>) -> tensor<4x4x!quant.uniform<i32:f32, 1.000000e-01>> {
                %0 = ttir.empty() : tensor<4x4x!quant.uniform<i32:f32, 1.000000e-01>>
                %1 = "ttir.quantize"(%arg0, %0) : (tensor<4x4xf32>, tensor<4x4x!quant.uniform<i32:f32, 1.000000e-01>>) -> tensor<4x4x!quant.uniform<i32:f32, 1.000000e-01>>
                return %1 : tensor<4x4x!quant.uniform<i32:f32, 1.000000e-01>>
            }
        }
        """,
            marks=pytest.mark.skip(reason="Not Supported Yet"),
        ),
        # Complex operations
        """
    module {
        func.func @conv_transpose2d_op(%arg0: tensor<3x8x8x256xf32>, %arg1: tensor<256x256x3x3xf32>, %arg2: tensor<1x1x1x256xf32>, %arg3: tensor<3x10x10x256xf32>) -> tensor<3x10x10x256xf32> {
            %0 = ttir.empty() : tensor<3x10x10x256xf32>
            %1 = "ttir.conv_transpose2d"(%arg0, %arg1, %arg2, %0) <{dilation = 1 : i32, groups = 1 : i32, output_padding = 0 : i32, padding = 0 : i32, stride = 1 : i32}> : (tensor<3x8x8x256xf32>, tensor<256x256x3x3xf32>, tensor<1x1x1x256xf32>, tensor<3x10x10x256xf32>) -> tensor<3x10x10x256xf32>
            return %1 : tensor<3x10x10x256xf32>
        }
    }
    """,
        # Layout operations
        pytest.param(
            """
        module {
            func.func @to_layout_op(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
                %0 = ttir.empty() : tensor<4x4xf32>
                %1 = ttir.to_layout %arg0, %0 : tensor<4x4xf32> into tensor<4x4xf32> -> tensor<4x4xf32>
                return %1 : tensor<4x4xf32>
            }
        }
        """,
            marks=pytest.mark.skip(reason="Not Supported Yet"),
        ),
        # Cache operations
        """
    module {
        func.func @update_cache_op(%arg0: tensor<1x32x64x512xf32>, %arg1: tensor<1x32x1x512xf32>, %arg2: tensor<1xi32>) -> tensor<1x32x64x512xf32> {
            %0 = ttir.empty() : tensor<1x32x64x512xf32>
            %1 = "ttir.update_cache"(%arg0, %arg1, %arg2) <{batch_offset = 0 : i32}> : (tensor<1x32x64x512xf32>, tensor<1x32x1x512xf32>, tensor<1xi32>) -> tensor<1x32x64x512xf32>
            return %1 : tensor<1x32x64x512xf32>
        }
    }
        """,
        # CCL (Collective Communication Library) operations
        pytest.param(
            """
        module {
            func.func @all_reduce_op(%arg0: tensor<4x4xf32>) -> tensor<4x4xf32> {
                %0 = ttir.empty() : tensor<4x4xf32>
                %1 = "ttir.all_reduce"(%arg0, %0) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<4x4xf32>, tensor<4x4xf32>) -> tensor<4x4xf32>
                return %1 : tensor<4x4xf32>
            }
        }
        """,
            marks=pytest.mark.skip(reason="Not Supported Yet"),
        ),
    ],
)
def test_mlir_to_ttir_builder(mlir_string):
    module, builder = mlir_to_ttir_builder(mlir_string)

    assert module is not None
    assert builder is not None

    for entry in module.body.operations:
        if isinstance(entry, func.FuncOp):
            assert entry.name is not None, f"Function name missing in {entry}"
            print(f"Function name: {entry.name}")
            assert len(entry.arguments) > 0, f"No arguments in function {entry.name}"
            print(f"Arguments: {entry.arguments}")
