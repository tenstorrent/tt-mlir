# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Utility functions for tt-alchemist
"""

import os
import sys
import subprocess
from pathlib import Path


def find_tt_mlir_tools():
    """
    Find the tt-mlir tools in the PATH

    Returns:
        dict: Dictionary mapping tool names to their paths
    """
    tools = {"ttmlir-opt": None, "ttmlir-translate": None}

    # Check if the tools are in the PATH
    for tool in tools:
        try:
            path = subprocess.check_output(
                ["which", tool], universal_newlines=True
            ).strip()
            tools[tool] = path
        except subprocess.CalledProcessError:
            pass

    return tools


def validate_environment():
    """
    Validate that the environment is set up correctly

    Returns:
        bool: True if the environment is valid, False otherwise
    """
    # Check if the tt-mlir tools are available
    tools = find_tt_mlir_tools()
    missing_tools = [tool for tool, path in tools.items() if path is None]

    if missing_tools:
        print(
            f"Error: The following tools are not in the PATH: {', '.join(missing_tools)}",
            file=sys.stderr,
        )
        print("Please make sure the tt-mlir environment is activated.", file=sys.stderr)
        return False

    return True


def get_templates_dir():
    """
    Get the path to the templates directory

    Returns:
        Path: Path to the templates directory
    """
    # First, check if the TTALCHEMIST_TEMPLATES_DIR environment variable is set
    if "TTALCHEMIST_TEMPLATES_DIR" in os.environ:
        return Path(os.environ["TTALCHEMIST_TEMPLATES_DIR"])

    # Next, check if we're running from the source directory
    source_dir = Path(__file__).parent.parent.parent
    if (source_dir / "templates").exists():
        return source_dir / "templates"

    # Finally, check the installed location
    import site

    for prefix in site.PREFIXES:
        templates_dir = Path(prefix) / "share" / "tt-alchemist" / "templates"
        if templates_dir.exists():
            return templates_dir

    # If we can't find the templates directory, use a default location
    return Path("/usr/local/share/tt-alchemist/templates")


def process_template(template_path, output_path, replacements):
    """
    Process a template file, replacing placeholders with values

    Args:
        template_path: Path to the template file
        output_path: Path to the output file
        replacements: Dictionary mapping placeholders to values

    Returns:
        bool: True if successful, False otherwise
    """
    try:
        # Read the template file
        with open(template_path, "r") as f:
            template = f.read()

        # Replace placeholders
        for placeholder, value in replacements.items():
            template = template.replace(placeholder, value)

        # Write the output file
        with open(output_path, "w") as f:
            f.write(template)

        return True
    except Exception as e:
        print(f"Error processing template: {e}", file=sys.stderr)
        return False
