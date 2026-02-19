#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Test script for the refactored chisel tool with TTNN-based goldens.

This script:
1. Compiles a TTNN MLIR file to a flatbuffer with embedded MLIR
2. Runs chisel with the flatbuffer-only workflow
3. Verifies that the refactored implementation works correctly
"""

import subprocess
import sys
from pathlib import Path
import tempfile
import shutil

def run_command(cmd, desc=""):
    """Run a shell command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{desc}")
    print(f"Command: {' '.join(str(c) for c in cmd)}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"STDERR:\n{result.stderr}")
        print(f"STDOUT:\n{result.stdout}")
        raise RuntimeError(f"Command failed with return code {result.returncode}")

    if result.stdout:
        print(f"STDOUT:\n{result.stdout}")

    return result

def main():
    # Get paths
    test_dir = Path(__file__).parent
    mlir_file = test_dir / "test_ttnn_add.mlir"

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        flatbuffer_file = tmpdir / "test_add.ttnn"
        output_dir = tmpdir / "chisel_output"
        report_file = tmpdir / "chisel_report.csv"

        print(f"\n{'#'*60}")
        print(f"# Testing Refactored Chisel Tool")
        print(f"{'#'*60}")
        print(f"MLIR file: {mlir_file}")
        print(f"Flatbuffer: {flatbuffer_file}")
        print(f"Output dir: {output_dir}")
        print(f"Report: {report_file}")

        # Step 1: Compile TTNN MLIR to flatbuffer with embedded MLIR
        print("\n\n" + "="*60)
        print("STEP 1: Compile TTNN MLIR to flatbuffer with embedded MLIR")
        print("="*60)

        try:
            # Import required modules
            from ttmlir.ir import Context, Module
            from ttmlir.passes import ttnn_to_flatbuffer_file
            import os

            # Read MLIR file
            with open(mlir_file, 'r') as f:
                mlir_source = f.read()

            print(f"✓ Read MLIR file ({len(mlir_source)} bytes)")

            # Parse MLIR module
            ctx = Context()
            ctx.load_all_available_dialects()
            module = Module.parse(mlir_source, ctx)

            print(f"✓ Parsed MLIR module")

            # Get system descriptor path (required for TTNN)
            system_desc = os.getenv("SYSTEM_DESC_PATH", "ttrt-artifacts/system_desc.ttsys")
            if not Path(system_desc).exists():
                print(f"WARNING: System descriptor not found at {system_desc}")
                print("Attempting to create default system descriptor...")
                # Try to run ttrt query --save-artifacts
                try:
                    subprocess.run(
                        ["ttrt", "query", "--save-artifacts"],
                        capture_output=True,
                        check=True
                    )
                    print(f"✓ Created system descriptor at {system_desc}")
                except Exception as e:
                    print(f"ERROR: Could not create system descriptor: {e}")
                    print("Please run: ttrt query --save-artifacts")
                    return 1

            # Create flatbuffer with embedded MLIR
            # The 4th parameter (moduleCache) is where we embed the MLIR source
            # Format: list of (name, source) tuples
            module_cache = [("ttnn", mlir_source)]

            print(f"✓ Embedding MLIR in flatbuffer (module_cache with {len(module_cache)} entries)")

            ttnn_to_flatbuffer_file(
                module,
                str(flatbuffer_file),
                {},  # goldenMap (empty)
                module_cache  # moduleCache with MLIR source
            )

            print(f"✓ Flatbuffer created: {flatbuffer_file}")
            print(f"  Size: {flatbuffer_file.stat().st_size} bytes")

            # Verify MLIR was embedded
            from ttrt.binary import load_binary_from_path, mlir_as_dict

            binary = load_binary_from_path(str(flatbuffer_file))
            mlir_dict = mlir_as_dict(binary)

            if mlir_dict.get('ttnn'):
                print(f"✓ Verified: MLIR embedded in flatbuffer")
                print(f"  Embedded MLIR size: {len(mlir_dict['ttnn'])} bytes")
                print(f"  Module cache keys: {list(mlir_dict.keys())}")
            else:
                print(f"✗ WARNING: MLIR not found in flatbuffer!")
                print(f"  Available keys: {list(mlir_dict.keys())}")
                print(f"  This may cause chisel to fail")

        except ImportError as e:
            print(f"ERROR: Failed to import required modules: {e}")
            print("Make sure you've activated the tt-xla environment")
            print("Run: cd /localdev/ndrakulic/tt-xla && source venv/activate")
            return 1
        except Exception as e:
            print(f"ERROR: Failed to compile TTNN to flatbuffer: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # Step 2: Run refactored chisel tool
        print("\n\n" + "="*60)
        print("STEP 2: Run refactored chisel tool")
        print("="*60)

        try:
            chisel_result = subprocess.run(
                [
                    sys.executable, "-m", "chisel.main",
                    "--flatbuffer-path", str(flatbuffer_file),
                    "--output-dir", str(output_dir),
                    "--report-path", str(report_file),
                    "--main-function", "main",
                    "--program-index", "0"
                ],
                capture_output=True,
                text=True,
                timeout=60
            )

            print(f"STDOUT:\n{chisel_result.stdout}")

            if chisel_result.returncode != 0:
                print(f"STDERR:\n{chisel_result.stderr}")
                print(f"\n✗ Chisel failed with return code {chisel_result.returncode}")
                return 1

            print(f"\n✓ Chisel executed successfully")

        except subprocess.TimeoutExpired:
            print("✗ Chisel timed out after 60 seconds")
            return 1
        except Exception as e:
            print(f"✗ Error running chisel: {e}")
            return 1

        # Step 3: Verify outputs
        print("\n\n" + "="*60)
        print("STEP 3: Verify outputs")
        print("="*60)

        # Check if report file was created
        if report_file.exists():
            print(f"✓ Report file created: {report_file}")
            print(f"  Size: {report_file.stat().st_size} bytes")

            # Print first few lines of report
            with open(report_file, 'r') as f:
                lines = f.readlines()[:10]
                print("\n  Report preview:")
                for line in lines:
                    print(f"    {line.rstrip()}")
        else:
            print(f"✗ Report file not found: {report_file}")
            return 1

        # Check if output directory was created
        if output_dir.exists():
            print(f"\n✓ Output directory created: {output_dir}")
            files = list(output_dir.rglob("*"))
            print(f"  Contains {len(files)} files/dirs")
            for f in files[:10]:  # Show first 10
                print(f"    - {f.relative_to(output_dir)}")

        print("\n\n" + "#"*60)
        print("# ALL TESTS PASSED!")
        print("#"*60)

        return 0

if __name__ == "__main__":
    sys.exit(main())
