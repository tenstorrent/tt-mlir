# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""
d2m-discover: Discover weakly connected components in MLIR modules.

This script walks an MLIR file and finds weakly connected components (WCCs)
of operations based on SSA value connectivity. It filters out components
of size 1 and operations matching an ignore list.
"""

import argparse
import io
import os
import sys
from collections import defaultdict

import ttmlir
import ttmlir.util
from ttmlir.ir import Context, Module, Location, InsertionPoint, Block, Operation
from ttmlir.dialects import func, ttir

# Default ignorelist file lives alongside this script.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OP_IGNORELIST_FILE = os.path.join(_SCRIPT_DIR, "op_ignorelist.txt")

# Ignorelist of operation names to ignore when building the graph.
# Operations matching these names will be excluded from analysis.
# This is the built-in default; it can be overridden at runtime via
# --op-ignorelist pointing to a text file (one op name per line).
OP_IGNORELIST = [
    "func.return",
    # ttcore
    "ttcore.load_cached",
    # ttnn
    "ttnn.get_device",
    "ttnn.paged_update_cache",
    "ttnn.softmax",
    "ttnn.assign",  # This op is used for ccl workaround
    "ttnn.all_gather",
    "ttnn.reduce_scatter",
    "ttnn.all_reduce",
    "ttnn.mesh_shard",
    "ttnn.point_to_point",
    # ttir
    "ttir.mesh_shard",
    "ttir.all_gather",
    "ttir.reduce_scatter",
    "ttir.all_reduce",
    "ttir.point_to_point",
    "ttir.all_to_all",
]


def load_op_ignorelist(filepath):
    """Load operation ignorelist from a text file.

    The file should contain one operation name per line.
    Lines starting with # and blank lines are ignored.
    """
    ops = []
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                ops.append(line)
    return ops


class UnionFind:
    """Union-Find data structure for efficient connected component discovery."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        """Find the representative of the set containing x with path compression."""
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x, y):
        """Union the sets containing x and y using union by rank."""
        px, py = self.find(x), self.find(y)
        if px == py:
            return
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1


def get_op_id(op):
    """Get a unique identifier for an operation."""
    return op.operation


def get_location_str(location):
    """Extract a human-readable location string from an MLIR location."""
    loc_str = str(location)
    # Location format is typically: loc("filename":line:col)
    # Extract the relevant parts
    if loc_str.startswith("loc("):
        loc_str = loc_str[4:-1]  # Remove loc( and )
    return loc_str


def process_func_op(func_op, uf, op_map):
    """Process a func.func operation and add its ops to the union-find structure."""
    for region in func_op.regions:
        for block in region:
            for op in block.operations:
                # Skip ignored operations
                if op.name in OP_IGNORELIST:
                    continue

                op_id = get_op_id(op)
                op_map[op_id] = op
                uf.find(op_id)  # Ensure the operation is in the union-find

                # Connect this operation to the operations that define its operands
                for operand in op.operands:
                    # Get the defining operation for this operand
                    # operand.owner gives us the operation that defines this value
                    defining_op = operand.owner
                    if defining_op is None:
                        # This is a block argument, not defined by an operation
                        continue

                    # owner can be a Block (for block arguments), skip those
                    if not hasattr(defining_op, "name"):
                        continue

                    # Skip if the defining operation is in the ignorelist
                    if defining_op.name in OP_IGNORELIST:
                        continue

                    defining_op_id = get_op_id(defining_op)

                    # Only connect if the defining op is in our set
                    # (it might be outside if we're not traversing into regions)
                    if defining_op_id in op_map:
                        uf.union(op_id, defining_op_id)


def find_connected_components(module):
    """
    Find weakly connected components in the MLIR module.

    Returns a list of components, where each component is a list of operations.
    Only components with more than one operation are returned.
    """
    uf = UnionFind()
    op_map = {}  # id -> operation

    # Collect all operations and build the union-find structure
    # Recurse into top-level func.func ops and func.func ops inside ttcore.device_module
    for entry in module.body.operations:
        op_name = entry.operation.name
        if op_name == "func.func" and (
            "sym_visibility" not in entry.attributes
            or entry.attributes["sym_visibility"] == "public"
        ):
            process_func_op(entry, uf, op_map)
        elif op_name == "ttcore.device_module":
            # Recurse into device_module to find nested func.func ops
            for region in entry.regions:
                for block in region:
                    for nested_op in block.operations:
                        assert nested_op.operation.name == "builtin.module"
                        for region in nested_op.operation.regions:
                            for block in region:
                                for nested_op in block.operations:
                                    if nested_op.operation.name == "func.func" and (
                                        "sym_visibility" not in nested_op.attributes
                                        or nested_op.attributes["sym_visibility"]
                                        == "public"
                                    ):
                                        process_func_op(nested_op, uf, op_map)

    # Group operations by their component representative
    components = defaultdict(list)
    for op_id, op in op_map.items():
        root = uf.find(op_id)
        components[root].append(op)

    # Filter out components of size 1
    return [ops for ops in components.values() if len(ops) > 1]


def print_components(components, file=None):
    """Print the discovered connected components.

    Args:
        components: List of connected components (each a list of ops).
        file: File-like object to write to (default: sys.stdout).
    """
    out = file or sys.stdout

    if not components:
        print("No connected components of size > 1 found.", file=out)
        return

    print(
        f"Found {len(components)} connected component(s) of size > 1:\n",
        file=out,
    )

    unique_ops = set()

    for i, ops in enumerate(components, 1):
        print(f"Component {i} ({len(ops)} operations):", file=out)
        for op in ops:
            loc_str = get_location_str(op.location)
            unique_ops.add(op.name)
            print(f"  - {op.name} @ {loc_str}", file=out)
        print(file=out)

    print("Unique operations across all components:", file=out)
    for op_name in sorted(unique_ops):
        print(f"  - {op_name}", file=out)


def topological_sort(ops):
    """Sort operations in topological order based on SSA dependencies."""
    op_set = set(get_op_id(op) for op in ops)
    op_by_id = {get_op_id(op): op for op in ops}

    # Build dependency graph within the component
    in_degree = {get_op_id(op): 0 for op in ops}
    dependents = defaultdict(list)

    for op in ops:
        op_id = get_op_id(op)
        for operand in op.operands:
            defining_op = operand.owner
            if defining_op is None or not hasattr(defining_op, "name"):
                continue
            def_id = get_op_id(defining_op)
            if def_id in op_set and def_id != op_id:
                in_degree[op_id] += 1
                dependents[def_id].append(op_id)

    # Kahn's algorithm
    queue = [op_id for op_id, deg in in_degree.items() if deg == 0]
    sorted_ops = []

    while queue:
        op_id = queue.pop(0)
        sorted_ops.append(op_by_id[op_id])
        for dep_id in dependents[op_id]:
            in_degree[dep_id] -= 1
            if in_degree[dep_id] == 0:
                queue.append(dep_id)

    return sorted_ops


def get_op_operands(op):
    """Get input and output operands for an operation (handles DPS convention)."""
    if "operandSegmentSizes" in op.attributes:
        segments = op.attributes["operandSegmentSizes"]
        if len(segments) == 2:
            # DPS-style: [inputs, inits]
            ins, outs = segments
            assert ins + outs == len(op.operands)
            return (list(op.operands[:ins]), list(op.operands[ins:]))
        # AttrSizedOperandSegments with 3+ groups (e.g. ttir.rms_norm: input, weight, bias).
        # All operands are inputs; no DPS inits.
        return (list(op.operands), [])
    elif ttmlir.util.is_dps(op):
        return (list(op.operands[:-1]), list(op.operands[-1:]))
    return (list(op.operands), [])


def emit_component_as_func(ops, func_name, ip, loc):
    """Emit a connected component as a standalone func.func."""
    sorted_ops = topological_sort(ops)
    op_set = set(get_op_id(op) for op in ops)

    # Find external inputs (operands defined outside the component)
    external_inputs = []  # List of (value, type)
    external_input_map = {}  # value -> index in external_inputs

    for op in sorted_ops:
        for operand in op.operands:
            defining_op = operand.owner
            # External if: no owner, owner is a Block, or owner not in component
            is_external = (
                defining_op is None
                or not hasattr(defining_op, "name")
                or get_op_id(defining_op) not in op_set
            )
            if is_external and operand not in external_input_map:
                external_input_map[operand] = len(external_inputs)
                external_inputs.append((operand, operand.type))

    # Find outputs (results of the last operation in topological order)
    # For simplicity, use all results of the final operation
    final_op = sorted_ops[-1]
    result_types = [result.type for result in final_op.results]

    # Create the function
    input_types = [t for _, t in external_inputs]
    entry = func.FuncOp(func_name, (input_types, result_types), ip=ip, loc=loc)
    entry_block = Block.create_at_start(entry.body, input_types)

    with InsertionPoint(entry_block) as block_ip, Location.unknown() as block_loc:
        # Map from original values to new values
        value_map = {}

        # Map external inputs to block arguments
        for orig_value, idx in external_input_map.items():
            value_map[orig_value] = entry_block.arguments[idx]

        # Clone each operation in topological order
        for op in sorted_ops:
            op_inputs, op_outputs = get_op_operands(op)

            # Remap input operands
            new_operands = []
            for operand in op_inputs:
                if operand in value_map:
                    new_operands.append(value_map[operand])
                else:
                    new_operands.append(operand)

            # Create empty tensors for DPS outputs
            for output in op_outputs:
                output_type = output.type
                empty_op = ttir.empty(
                    output_type.shape,
                    output_type.element_type,
                    encoding=output_type.encoding,
                    ip=block_ip,
                    loc=block_loc,
                )
                new_operands.append(empty_op)

            # Clone the operation
            attrs = {attr.name: attr.attr for attr in op.attributes}
            new_op = Operation.create(
                name=op.name,
                results=[r.type for r in op.results],
                operands=new_operands,
                attributes=attrs,
                successors=op.successors,
                regions=len(op.regions),
                loc=block_loc,
                ip=block_ip,
            )

            # Map results
            for orig_result, new_result in zip(op.results, new_op.results):
                value_map[orig_result] = new_result

        # Create return with the final operation's results
        final_results = [value_map[r] for r in final_op.results]
        Operation.create(
            name="func.return",
            results=[],
            operands=final_results,
            loc=block_loc,
            ip=block_ip,
        )


def emit_components_as_module(components, ctx):
    """Emit all connected components as a new MLIR module with func.func ops."""
    out_module = Module.create(Location.unknown(ctx))
    with InsertionPoint(out_module.body) as ip, Location.unknown() as loc:
        for i, ops in enumerate(components, 1):
            func_name = f"component_{i}"
            emit_component_as_func(ops, func_name, ip, loc)
    return out_module


def main():
    parser = argparse.ArgumentParser(
        description="d2m-discover: Discover weakly connected components in MLIR modules"
    )
    parser.add_argument("mlir", type=str, help="Path to the MLIR file")
    parser.add_argument(
        "--emit-funcs",
        action="store_true",
        help="Emit components as standalone func.func ops (outputs MLIR)",
    )
    parser.add_argument(
        "--op-ignorelist",
        type=str,
        default=None,
        help=(
            "Path to a text file with operation names to ignore (one per line). "
            f"Default: {_DEFAULT_OP_IGNORELIST_FILE}"
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to write output to (default: print to stdout)",
    )
    args = parser.parse_args()

    # --- Logging helper (all progress goes to stderr) ---
    def log(msg):
        print(msg, file=sys.stderr, flush=True)

    # --- Validate input ---
    if not os.path.isfile(args.mlir):
        log(f"Error: file not found: {args.mlir}")
        sys.exit(1)

    # --- Banner ---
    log("=" * 60)
    log("d2m-discover: Weakly connected component discovery")
    log("=" * 60)
    file_size_mb = os.path.getsize(args.mlir) / (1024 * 1024)
    log(f"  Input file:    {args.mlir} ({file_size_mb:.1f} MB)")

    # Override the global ignorelist if a custom file is provided.
    global OP_IGNORELIST
    if args.op_ignorelist:
        OP_IGNORELIST = load_op_ignorelist(args.op_ignorelist)
        log(f"  Op ignorelist: {args.op_ignorelist} ({len(OP_IGNORELIST)} ops)")
    else:
        log(f"  Op ignorelist: built-in default ({len(OP_IGNORELIST)} ops)")

    mode_label = (
        "emit-funcs (MLIR snippet output)"
        if args.emit_funcs
        else "component overview (text output)"
    )
    log(f"  Mode:          {mode_label}")
    log(f"  Output:        {args.output or 'stdout'}")
    log("")

    # --- Step 1: Parse ---
    log("[1/3] Parsing MLIR module ...")
    with Context() as ctx, open(args.mlir, "r") as mlir_fd:
        ctx.allow_unregistered_dialects = True
        try:
            module = Module.parse(mlir_fd.read())
        except Exception as exc:
            log(f"  ERROR: Failed to parse MLIR: {exc}")
            sys.exit(1)
        log("  Parsed successfully.")

        # --- Step 2: Discover ---
        log("[2/3] Finding weakly connected components ...")
        components = find_connected_components(module)
        total_ops = sum(len(c) for c in components)
        log(
            f"  Found {len(components)} component(s) "
            f"with {total_ops} total operations."
        )

        # --- Step 3: Generate output ---
        log("[3/3] Generating output ...")
        if args.emit_funcs:
            out_module = emit_components_as_module(components, ctx)
            output_text = str(out_module)
            log(f"  Emitted {len(components)} func.func op(s) as MLIR module.")
        else:
            buf = io.StringIO()
            print_components(components, file=buf)
            output_text = buf.getvalue()
            log("  Generated component overview text.")

    # --- Write output ---
    if args.output:
        try:
            out_dir = os.path.dirname(args.output)
            if out_dir:
                os.makedirs(out_dir, exist_ok=True)
            with open(args.output, "w") as f:
                f.write(output_text)
        except OSError as exc:
            log(f"  ERROR: Failed to write output: {exc}")
            sys.exit(1)
        log(f"\nDone. Output written to: {args.output}")
    else:
        print(output_text, end="")
        log(f"\nDone. Output written to stdout.")


if __name__ == "__main__":
    main()
