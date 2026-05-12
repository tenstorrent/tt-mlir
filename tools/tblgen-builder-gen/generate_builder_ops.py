#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
TableGen-based Builder Op Generator

This script generates Python builder functions from MLIR TableGen (.td) definitions.
It parses the TableGen JSON output and generates plugin-compatible builder methods.

Usage:
    # Generate JSON from TableGen
    mlir-tblgen --dump-json include/ttmlir/Dialect/TTIR/IR/TTIROps.td > ttir_ops.json

    # Generate Python builder code
    python generate_builder_ops.py ttir_ops.json --output dialects/ttir_generated.py
"""

import json
import argparse
from typing import Dict, List, Optional, Any
from dataclasses import dataclass


@dataclass
class OpArgument:
    """Represents an operation argument/operand."""

    name: str
    type: str
    description: str = ""


@dataclass
class OpResult:
    """Represents an operation result."""

    name: str
    type: str


@dataclass
class OpInfo:
    """Parsed information about an MLIR operation."""

    name: str  # e.g., "TTIR_SigmoidOp"
    mnemonic: str  # e.g., "sigmoid"
    dialect: str  # e.g., "ttir"
    class_name: str  # e.g., "SigmoidOp"
    arguments: List[OpArgument]
    results: List[OpResult]
    summary: str
    description: str
    base_class: str  # e.g., "TTIR_ElementwiseUnaryOp"
    traits: List[str]


def parse_tablegen_json(json_data: Dict) -> List[OpInfo]:
    """
    Parse TableGen JSON output to extract operation information.

    Note: This is a simplified parser. Real TableGen JSON is complex
    and would need more sophisticated parsing.
    """
    ops = []

    # TableGen JSON structure varies, this is a placeholder
    # In practice, you'd use mlir-tblgen --dump-json or similar

    return ops


def parse_td_file_simple(td_content: str) -> List[OpInfo]:
    """
    Simple parser for .td files to extract basic op information.

    This is a proof-of-concept parser that handles simple cases.
    For production, use mlir-tblgen's JSON output or Python bindings.
    """
    ops = []
    lines = td_content.split("\n")

    i = 0
    while i < len(lines):
        line = lines[i].strip()

        # Look for op definitions: def TTIR_SigmoidOp: TTIR_ElementwiseUnaryOp<...>
        if line.startswith("def TTIR_") and ":" in line:
            op_info = parse_op_definition(lines, i)
            if op_info:
                ops.append(op_info)

        i += 1

    return ops


def parse_op_definition(lines: List[str], start_idx: int) -> Optional[OpInfo]:
    """Parse a single op definition from .td file."""
    line = lines[start_idx].strip()

    # Extract: def TTIR_SigmoidOp: TTIR_ElementwiseUnaryOp<"sigmoid">
    if not line.startswith("def TTIR_"):
        return None

    parts = line.split(":")
    if len(parts) < 2:
        return None

    name = parts[0].replace("def ", "").strip()
    base_and_mnemonic = parts[1].strip()

    # Extract base class and mnemonic
    base_class = base_and_mnemonic.split("<")[0].strip()
    mnemonic = None
    if "<" in base_and_mnemonic and ">" in base_and_mnemonic:
        mnemonic = base_and_mnemonic.split("<")[1].split(">")[0].strip('"')

    # Extract summary and description
    summary = ""
    description = ""
    i = start_idx + 1
    while i < len(lines) and not lines[i].strip().startswith("def "):
        line = lines[i].strip()
        if "let summary" in line:
            summary = extract_string_literal(line)
        elif "let description" in line:
            description = extract_description_block(lines, i)
        i += 1

    # Determine arguments based on base class
    arguments = []
    results = []

    if "ElementwiseUnaryOp" in base_class:
        arguments = [OpArgument("input", "AnyRankedTensor", "Input tensor")]
        results = [OpResult("result", "AnyRankedTensor")]
    elif "ElementwiseBinaryOp" in base_class:
        arguments = [
            OpArgument("lhs", "AnyRankedTensor", "Left operand"),
            OpArgument("rhs", "AnyRankedTensor", "Right operand"),
        ]
        results = [OpResult("result", "AnyRankedTensor")]

    return OpInfo(
        name=name,
        mnemonic=mnemonic or name.replace("TTIR_", "").replace("Op", "").lower(),
        dialect="ttir",
        class_name=name.replace("TTIR_", ""),
        arguments=arguments,
        results=results,
        summary=summary,
        description=description,
        base_class=base_class,
        traits=[],
    )


def extract_string_literal(line: str) -> str:
    """Extract string literal from a line like: let summary = "Eltwise sigmoid.";"""
    if '"' not in line:
        return ""
    parts = line.split('"')
    if len(parts) >= 2:
        return parts[1]
    return ""


def extract_description_block(lines: List[str], start_idx: int) -> str:
    """Extract multi-line description block."""
    description_lines = []
    i = start_idx
    in_block = False

    while i < len(lines):
        line = lines[i]
        if "[{" in line:
            in_block = True
        if in_block:
            description_lines.append(line)
        if "}]" in line:
            break
        i += 1

    return "\n".join(description_lines)


def generate_builder_method(op: OpInfo, template: str = "unary") -> str:
    """
    Generate Python builder method code for an operation.

    Args:
        op: Operation information
        template: Template to use (unary, binary, etc.)
    """
    if template == "unary":
        return generate_unary_op(op)
    elif template == "binary":
        return generate_binary_op(op)
    else:
        return generate_generic_op(op)


def generate_unary_op(op: OpInfo) -> str:
    """Generate builder method for unary operations like sigmoid, relu, cos."""

    method_name = op.mnemonic
    op_class = f"{op.dialect}.{op.class_name}"

    code = f'''
    @tag({op_class})
    def {method_name}(
        self,
        builder,
        in0: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        """
        {op.summary}

        Args:
            builder: Builder instance
            in0: Input operand
            output_type: Optional output dtype (defaults to input dtype)
            loc: Optional location string for debugging
            unit_attrs: Optional unit attributes to add to operation

        Returns:
            Operation result
        """
        op_class = {op_class}

        # Determine output type
        if output_type is None:
            mlir_output_type = builder.get_type(in0)
        else:
            mlir_output_type = builder._get_type_from_torch_dtype(output_type)

        # Get golden tensor and compute output
        input0 = builder._get_golden_tensor(in0)
        op_golden_function = get_golden_function(op_class)
        golden_output = op_golden_function(input0, mlir_output_type)
        result = RankedTensorType.get(golden_output.shape, mlir_output_type)

        # Create location
        if loc is None:
            loc = Location.unknown(builder.context)
        else:
            loc = Location.name(loc)

        # Create MLIR operation
        op = op_class(result, in0, loc=loc)
        op_result = op.result

        # Add unit attributes if specified
        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(builder.context)

        # Store golden tensor
        builder._set_golden_tensor(op_result, golden_output)

        return op_result

    @parse({op_class})
    def {method_name}_parser(
        self,
        builder,
        old_op: {op_class},
        global_dict: Dict[Operand, Operand],
    ) -> Tuple[Operation, Dict[OpResult, OpResult]]:
        """Parse {method_name} operation from existing MLIR."""
        op_class = {op_class}
        in0 = global_dict[old_op.input]
        result = old_op.result.type

        # Create new operation
        new_op = op_class(result, in0, loc=old_op.location)
        new_op_result = new_op.result

        # Compute golden output
        input0 = builder._get_golden_tensor(in0)
        op_golden_function = get_golden_function(op_class)
        golden_output = op_golden_function(input0, result.element_type)
        builder._set_golden_tensor(new_op_result, golden_output)

        # Map old result to new result
        op_map_dictionary = {{}}
        op_map_dictionary[old_op.result] = new_op_result
        return new_op, op_map_dictionary

    @split({op_class})
    def {method_name}_split(
        self,
        builder,
        old_op: {op_class},
    ) -> Tuple[Module, "Builder"]:
        """Split {method_name} operation into separate module."""
        op_class = {op_class}

        old_ctx = old_op.context
        old_loc = Location.unknown(old_ctx)
        with old_ctx, old_loc:
            {method_name}_module = Module.create()

            # Create new builder for split module
            from builder_prototype.builder import Builder
            {method_name}_builder = Builder(
                old_ctx, old_loc,
                mesh_name=builder._mesh_name,
                mesh_dict=builder._mesh_dict
            )
            # Register same dialects as parent
            for dialect_name, plugin in builder._plugins.items():
                {method_name}_builder.register_dialect(dialect_name, plugin)

            op_input_types = [old_op.input.type]

            with InsertionPoint({method_name}_module.body):
                ordered_inputs = []
                ordered_outputs = []

                @func.func(*op_input_types, name="{method_name}_module")
                def decorated_func(*inputs):
                    in0 = inputs[0]
                    result = old_op.result.type

                    new_op = op_class(result, in0, loc=old_op.location)
                    new_op_result = new_op.result

                    input0 = builder._get_golden_tensor(old_op.input)
                    old_op_result = builder._get_golden_tensor(old_op.result)
                    {method_name}_builder._set_golden_tensor(new_op_result, old_op_result)
                    {method_name}_builder._set_golden_tensor(in0, input0)
                    ordered_inputs.append(in0)
                    ordered_outputs.append(new_op_result)

                    return new_op

                new_func_op = decorated_func.func_op
                {method_name}_builder._func_ops_generated[new_func_op] = [
                    ordered_inputs,
                    ordered_outputs,
                ]

        return {method_name}_module, {method_name}_builder
'''

    return code


def generate_binary_op(op: OpInfo) -> str:
    """Generate builder method for binary operations like add, mul, etc."""

    method_name = op.mnemonic
    op_class = f"{op.dialect}.{op.class_name}"

    code = f'''
    @tag({op_class})
    def {method_name}(
        self,
        builder,
        lhs: Operand,
        rhs: Operand,
        output_type: Optional[torch.dtype] = None,
        loc: Optional[str] = None,
        unit_attrs: Optional[List[str]] = None,
    ) -> OpResult:
        """
        {op.summary}

        Args:
            builder: Builder instance
            lhs: Left operand
            rhs: Right operand
            output_type: Optional output dtype
            loc: Optional location string
            unit_attrs: Optional unit attributes

        Returns:
            Operation result
        """
        op_class = {op_class}

        # Determine output type
        if output_type is None:
            mlir_output_type = builder.get_type(lhs)
        else:
            mlir_output_type = builder._get_type_from_torch_dtype(output_type)

        # Get golden tensors and compute output
        input_lhs = builder._get_golden_tensor(lhs)
        input_rhs = builder._get_golden_tensor(rhs)
        op_golden_function = get_golden_function(op_class)
        golden_output = op_golden_function(input_lhs, input_rhs, mlir_output_type)
        result = RankedTensorType.get(golden_output.shape, mlir_output_type)

        # Create location
        if loc is None:
            loc = Location.unknown(builder.context)
        else:
            loc = Location.name(loc)

        # Create MLIR operation
        op = op_class(result, lhs, rhs, loc=loc)
        op_result = op.result

        # Add unit attributes if specified
        if unit_attrs is not None:
            for attr_name in unit_attrs:
                op.operation.attributes[attr_name] = UnitAttr.get(builder.context)

        # Store golden tensor
        builder._set_golden_tensor(op_result, golden_output)

        return op_result
'''

    return code


def generate_generic_op(op: OpInfo) -> str:
    """Generate builder method for operations that don't fit standard templates."""
    return f"# TODO: Generate generic op for {op.name}\n"


def generate_plugin_file(ops: List[OpInfo], dialect: str = "ttir") -> str:
    """Generate complete Python plugin file with all operations."""

    dialect_upper = dialect.upper()
    dialect_title = dialect.capitalize()

    header = f'''# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
{dialect_title} Dialect Plugin - GENERATED CODE

This file is automatically generated from TableGen definitions.
DO NOT EDIT MANUALLY - changes will be overwritten.

Generated from: include/ttmlir/Dialect/{dialect_upper}/IR/{dialect_upper}Ops.td
"""

from typing import List, Optional, Dict, Tuple
import torch

from ttmlir.ir import *
from ttmlir.dialects import func, {dialect}, ttcore
from ttmlir.passes import DataType
from golden import get_golden_function

from builder_prototype.dialect_plugin import DialectPlugin, tag, parse, split


class {dialect_title}Plugin(DialectPlugin):
    """
    {dialect_title} dialect plugin - AUTO-GENERATED

    Provides {dialect_upper} operations with plugin architecture.
    Generated from TableGen operation definitions.
    """

    def register_ops(self, builder) -> None:
        """Register {dialect_upper} operations with the builder."""
        # Operations are automatically discovered via decorators
        pass

    def create_tensor_encoding(
        self,
        builder,
        tensor_type: RankedTensorType,
        memory_space: str = "system"
    ) -> Attribute:
        """Create {dialect_upper} tensor encoding attribute."""
        # {dialect_upper}-specific tensor encoding
        grid = ttcore.ir.GridAttr.get(builder.context, [0, 0], [0, 0])

        memory_space_enum = ttcore.ir.MemorySpace.System
        if memory_space == "device":
            memory_space_enum = ttcore.ir.MemorySpace.DeviceDRAM
        elif memory_space == "l1":
            memory_space_enum = ttcore.ir.MemorySpace.DeviceL1

        memory_attr = ttcore.ir.MemorySpaceAttr.get(builder.context, memory_space_enum)

        memory_config = ttcore.ir.MemoryConfigEncodingAttr.get(
            builder.context,
            memory_attr,
            ttcore.ir.TensorMemoryLayoutAttr.get(
                builder.context, ttcore.ir.TensorMemoryLayout.Interleaved
            ),
            ttcore.ir.BufferTypeAttr.get(
                builder.context, ttcore.ir.BufferType.DRAM
            ),
            None,
        )

        return memory_config
'''

    # Generate all operation methods
    operations_code = "\n    # ----- Generated Operations -----\n"

    for op in ops:
        if "ElementwiseUnaryOp" in op.base_class:
            operations_code += generate_unary_op(op)
        elif "ElementwiseBinaryOp" in op.base_class:
            operations_code += generate_binary_op(op)
        else:
            operations_code += f"\n    # TODO: Generate {op.name}\n"

    return header + operations_code


def main():
    parser = argparse.ArgumentParser(
        description="Generate Python builder functions from TableGen definitions"
    )
    parser.add_argument("input", help="Input .td file or JSON file from mlir-tblgen")
    parser.add_argument(
        "--output", default="ttir_generated.py", help="Output Python file"
    )
    parser.add_argument(
        "--dialect", default="ttir", help="Dialect name (default: ttir)"
    )
    parser.add_argument(
        "--ops", nargs="*", help="Specific ops to generate (default: all)"
    )

    args = parser.parse_args()

    # Read input file
    with open(args.input, "r") as f:
        content = f.read()

    # Parse operations
    if args.input.endswith(".json"):
        data = json.loads(content)
        ops = parse_tablegen_json(data)
    else:
        # Simple .td file parsing
        ops = parse_td_file_simple(content)

    # Filter ops if specified
    if args.ops:
        ops = [op for op in ops if op.mnemonic in args.ops]

    print(f"Found {len(ops)} operations to generate")
    for op in ops:
        print(f"  - {op.name} ({op.mnemonic})")

    # Generate plugin file
    plugin_code = generate_plugin_file(ops, args.dialect)

    # Write output
    with open(args.output, "w") as f:
        f.write(plugin_code)

    print(f"\nGenerated {args.output}")
    print(f"Total operations: {len(ops)}")


if __name__ == "__main__":
    main()
