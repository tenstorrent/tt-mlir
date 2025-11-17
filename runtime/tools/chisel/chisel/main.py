# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import datetime
import os
import argparse

from chisel.core.compile_pipeline import chisel_pipeline
from chisel.core.context import ChiselContext
from chisel.core.enums import ExecutionType
from ttmlir.ir import Operation
from ttmlir.passes import ttnn_to_flatbuffer_file


def parse_arguments():
    """Parse command line arguments for chisel execution."""
    parser = argparse.ArgumentParser(
        description="Chisel TTIR/TTNN execution and comparison tool",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output paths
    parser.add_argument(
        "--input-file",
        "-i",
        type=Path,
        default=Path("runtime/tools/chisel/test/mlir/test_fusion.mlir"),
        help="Path to input TTIR file",
    )

    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("runtime/tools/chisel/test/mlir/output"),
        help="Output directory for results",
    )

    parser.add_argument(
        "--tensor-folder",
        type=Path,
        default=Path("runtime/tools/chisel/test/mlir/tensors"),
        help="Directory containing input tensor files",
    )

    parser.add_argument(
        "--flatbuffer-path",
        type=Path,
        default=Path("runtime/tools/chisel/test/mlir/fb.ttnn"),
        help="Path to flatbuffer file",
    )

    parser.add_argument(
        "--report-path",
        type=Path,
        help="Path for the output report CSV file (default: auto-generated with timestamp)",
    )

    # Execution parameters
    parser.add_argument(
        "--main-function",
        "-f",
        type=str,
        default="main",
        help="Name of the main function to execute",
    )

    parser.add_argument(
        "--program-index", type=int, default=0, help="Program index for execution"
    )

    # Input generation options
    parser.add_argument(
        "--use-random-inputs",
        action="store_true",
        default=True,
        help="Generate random inputs instead of loading from disk",
    )

    parser.add_argument(
        "--load-inputs-from-disk",
        action="store_true",
        help="Load inputs from tensor folder instead of generating random inputs",
    )

    # Skip operation configuration
    parser.add_argument(
        "--skip-op-pattern",
        type=str,
        help="Pattern to match operations that should be skipped (e.g., '%%6 = \"ttnn.matmul\"')",
    )

    # Dump options
    parser.add_argument(
        "--dump-ttir",
        action="store_true",
        default=False,
        help="Dump TTIR module to chisel_ttir.mlir file",
    )

    parser.add_argument(
        "--dump-ttnn",
        action="store_true",
        default=False,
        help="Dump TTNN module to chisel_ttnn.mlir file",
    )

    return parser.parse_args()


def create_skip_op_function(pattern=None):
    """Create a skip operation function based on the provided pattern."""
    if pattern is None:
        return lambda op: False

    def should_skip_op(op):
        try:
            op_asm = op.get_asm(enable_debug_info=True)
            return pattern in op_asm
        except Exception:
            return False

    return should_skip_op


def main():
    """
    This script compiles TTIR files to generate both TTIR (golden) and TTNN (device)
    modules using the chisel pipeline, then executes and compares them.

    Run with --help to see all available options.
    """
    args = parse_arguments()

    # Validate input file exists
    if not args.input_file.exists() and args.input_file.endswith(".mlir"):
        raise FileNotFoundError(f"Input file does not exist: {args.input_file}")

    # Ensure output directory exists and is writable
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        raise PermissionError(
            f"Cannot create directory for output: {args.output_dir}: {e}"
        )

    # Validate report path directory is writable
    try:
        args.report_path.parent.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        raise PermissionError(
            f"Cannot create directory for report file {args.report_path}: {e}"
        )

    print(f"Compiling TTIR from: {args.input_file}")

    # Use chisel_pipeline to compile TTIR to both golden and device modules
    try:
        ttir_module, ttnn_module = chisel_pipeline(
            args.input_file, args.dump_ttir, args.dump_ttnn
        )
    except Exception as e:
        raise RuntimeError(f"Failed to compile TTIR pipeline: {e}")

    ttnn_to_flatbuffer_file(ttnn_module, str(args.flatbuffer_path), {}, {})

    print("TTIR compilation completed successfully")

    should_skip_op = create_skip_op_function(args.skip_op_pattern)

    chisel_context = ChiselContext(
        ttir_module=ttir_module,
        ttnn_module=ttnn_module,
        output_dir=args.output_dir,
        report_path=args.report_path,
        main_fn=args.main_function,
        program_index=args.program_index,
        flatbuffer_path=args.flatbuffer_path,
        should_skip_op=should_skip_op,
    )
    chisel_context.bind_callbacks()

    if args.load_inputs_from_disk:
        input_paths = discover_input_tensor_paths(args.tensor_folder)
        if input_paths:
            print(f"Loading {len(input_paths)} inputs from disk:")
            print(*input_paths, sep="\n")
            try:
                chisel_context.load_inputs_from_disk(input_paths)
            except Exception as e:
                print(f"Warning: Failed to load inputs from disk: {e}")
                print("Falling back to random input generation")
                chisel_context.generate_random_inputs()
        else:
            print("No valid tensor files found, using random inputs instead")
            chisel_context.generate_random_inputs()
    else:
        print("Generating random inputs")
        chisel_context.generate_random_inputs()

    chisel_context.registry.load_all_ops()

    chisel_context.run()


def discover_input_tensor_paths(tensor_folder):
    """
    Discover and validate input tensor paths from the specified folder.

    Args:
        tensor_folder (Path): Directory to search for tensor files

    Returns:
        list[Path]: List of valid tensor file paths, sorted for consistent ordering

    Raises:
        FileNotFoundError: If tensor folder doesn't exist
        PermissionError: If tensor folder is not readable
    """
    if not tensor_folder.exists():
        raise FileNotFoundError(f"Tensor folder does not exist: {tensor_folder}")

    input_paths = []
    try:
        input_paths = [f for f in tensor_folder.iterdir() if f.is_file()]
    except OSError as e:
        raise OSError(f"Failed to read tensor folder {tensor_folder}: {e}")

    return sorted(input_paths, key=lambda p: int(p.stem))


if __name__ == "__main__":
    main()
