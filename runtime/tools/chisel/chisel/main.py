# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import argparse

from ttmlir.ir import Operation


def parse_arguments():
    """Parse command line arguments for chisel execution."""
    parser = argparse.ArgumentParser(
        description="Chisel: TTNN differential debugging tool (CPU golden vs device execution)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
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

    # Input/Output paths
    parser.add_argument(
        "--flatbuffer-path",
        "-f",
        type=Path,
        required=True,
        help="Path to flatbuffer file (MLIR will be extracted automatically)",
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
    TTNN differential debugging tool that compares CPU golden execution vs device hardware execution.

    This tool extracts TTNN MLIR from a flatbuffer, executes operations on both CPU (golden)
    and device (hardware), and compares the outputs to identify discrepancies.

    Run with --help to see all available options.
    """
    args = parse_arguments()

    # Validate flatbuffer exists
    if not args.flatbuffer_path.exists():
        raise FileNotFoundError(f"Flatbuffer not found: {args.flatbuffer_path}")

    # Ensure output directory exists and is writable
    try:
        args.output_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        raise PermissionError(
            f"Cannot create directory for output: {args.output_dir}: {e}"
        )

    # Validate report path directory is writable
    if args.report_path:
        try:
            args.report_path.parent.mkdir(parents=True, exist_ok=True)
        except (PermissionError, OSError) as e:
            raise PermissionError(
                f"Cannot create directory for report file {args.report_path}: {e}"
            )

    print(f"[Chisel] Loading flatbuffer from: {args.flatbuffer_path}")

    # Create skip operation function
    should_skip_op = create_skip_op_function(args.skip_op_pattern)

    # Create global context (lightweight - no MLIR loading yet)
    from chisel.core.context import setup_chisel, bind_chisel_callbacks

    chisel_context = setup_chisel(
        output_dir=args.output_dir,
        report_path=args.report_path,
        main_fn=args.main_function,
        program_index=args.program_index,
        flatbuffer_path=args.flatbuffer_path,
        should_skip_op=should_skip_op,
    )

    # Register chisel callbacks using DebugHooks
    bind_chisel_callbacks()

    # Handle input tensor setup (if needed)
    # Note: For now, we'll rely on the flatbuffer's built-in inputs
    # TODO: Support loading inputs from disk if needed
    if args.load_inputs_from_disk:
        print("[Warning] Loading inputs from disk not yet supported in flatbuffer-only mode")
        print("[Warning] Will use flatbuffer's built-in inputs or runtime initialization")

    # Run device execution with callbacks
    # MLIR will be extracted from flatbuffer on first preop callback
    from ttrt.runtime import submit
    from ttrt.binary import load_binary_from_path

    print("[Chisel] Loading binary and starting execution...")
    binary = load_binary_from_path(str(args.flatbuffer_path))

    # Execute with registered callbacks - context initialized lazily on first preop
    result = submit(binary, args.program_index, [], [])

    print(f"[Chisel] Execution completed")
    print(f"[Chisel] Report written to: {args.report_path}")

    return result


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
