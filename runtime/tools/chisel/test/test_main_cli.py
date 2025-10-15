# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import subprocess
import tempfile
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "ttir_path, function_name",
    [
        ("runtime/tools/chisel/test/mlir/test_fusion.mlir", "transpose_matmul"),
    ],
)
def test_chisel_cli_main(ttir_path: str, function_name: str):
    """
    End-to-end test for chisel CLI tool (main.py).

    This test validates that the CLI entrypoint works correctly, catching
    integration issues that library-only tests might miss (e.g., function
    signature mismatches in the main.py script).
    """
    ttir_path = Path(ttir_path)
    assert ttir_path.exists(), f"Test MLIR file not found: {ttir_path}"

    # Create temporary directory for outputs
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        report_path = tmpdir_path / "report.csv"
        output_dir = tmpdir_path / "output"
        flatbuffer_path = tmpdir_path / "fb.ttnn"

        # Build the command to run chisel main.py
        cmd = [
            "python",
            "runtime/tools/chisel/chisel/main.py",
            "--input-file",
            str(ttir_path),
            "--main-function",
            function_name,
            "--report-path",
            str(report_path),
            "--output-dir",
            str(output_dir),
            "--flatbuffer-path",
            str(flatbuffer_path),
            "--use-random-inputs",
        ]

        # Run the CLI command
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=Path.cwd(),  # Run from repo root
        )

        # Check that the command succeeded
        assert result.returncode == 0, (
            f"Chisel CLI failed with return code {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )

        # Verify output files were created
        assert (
            flatbuffer_path.exists()
        ), f"Flatbuffer file not created: {flatbuffer_path}"
        assert report_path.exists(), f"Report file not created: {report_path}"
        assert output_dir.exists(), f"Output directory not created: {output_dir}"

        # Verify report has content (should have at least header + some operations)
        with open(report_path, "r") as f:
            lines = f.readlines()
            assert len(lines) > 1, "Report file is empty or has no data rows"

        # Verify flatbuffer has content
        assert flatbuffer_path.stat().st_size > 0, "Flatbuffer file is empty"

        print(f"CLI test passed successfully for function: {function_name}")
        print(f"Report rows: {len(lines)}")
        print(f"Flatbuffer size: {flatbuffer_path.stat().st_size} bytes")

