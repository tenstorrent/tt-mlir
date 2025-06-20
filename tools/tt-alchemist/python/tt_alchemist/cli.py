# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
"""
Command-line interface for tt-alchemist
"""

import os
import sys
import click
from . import TTAlchemist, OptimizationLevel, BuildFlavor, HardwareTarget


@click.group()
@click.version_option()
def main():
    """tt-alchemist: A user-friendly abstraction layer for tt-mlir"""
    pass


@main.command("model-to-cpp")
@click.argument("input_file", type=click.Path(exists=True))
@click.option("--output", "-o", required=True, help="Output directory")
@click.option(
    "--opt-level",
    type=click.Choice(["minimal", "normal", "aggressive"]),
    default="normal",
    help="Optimization level",
)
def model_to_cpp(input_file, output, opt_level):
    """Convert a model to C++"""
    # Create the output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)

    # Create the TTAlchemist instance
    alchemist = TTAlchemist()

    # Map the optimization level
    opt_level_map = {
        "minimal": OptimizationLevel.MINIMAL,
        "normal": OptimizationLevel.NORMAL,
        "aggressive": OptimizationLevel.AGGRESSIVE,
    }

    # Create the conversion config
    config = {"opt_level": opt_level_map[opt_level], "output_dir": output}

    # Convert the model to C++
    click.echo(f"Converting {input_file} to C++ with {opt_level} optimization...")
    if not alchemist.model_to_cpp(input_file, config):
        click.echo(f"Error: {alchemist.get_last_error()}", err=True)
        sys.exit(1)

    click.echo(f"Model converted successfully to {output}")


@main.command("build")
@click.argument("model_dir", type=click.Path(exists=True))
@click.option(
    "--flavor",
    type=click.Choice(["release", "debug", "profile"]),
    default="release",
    help="Build flavor",
)
@click.option(
    "--target",
    type=click.Choice(["grayskull", "wormhole", "blackhole"]),
    default="grayskull",
    help="Hardware target",
)
def build(model_dir, flavor, target):
    """Build a generated solution"""
    # Create the TTAlchemist instance
    alchemist = TTAlchemist()

    # Map the build flavor
    flavor_map = {
        "release": BuildFlavor.RELEASE,
        "debug": BuildFlavor.DEBUG,
        "profile": BuildFlavor.PROFILE,
    }

    # Map the hardware target
    target_map = {
        "grayskull": HardwareTarget.GRAYSKULL,
        "wormhole": HardwareTarget.WORMHOLE,
        "blackhole": HardwareTarget.BLACKHOLE,
    }

    # Create the build config
    config = {"flavor": flavor_map[flavor], "target": target_map[target]}

    # Build the solution
    click.echo(f"Building {model_dir} with {flavor} flavor for {target}...")
    if not alchemist.build_solution(model_dir, config):
        click.echo(f"Error: {alchemist.get_last_error()}", err=True)
        sys.exit(1)

    click.echo(f"Solution built successfully")


@main.command("run")
@click.argument("model_dir", type=click.Path(exists=True))
@click.option("--input", "-i", help="Input file")
@click.option("--output", "-o", help="Output file")
def run(model_dir, input, output):
    """Run a built solution"""
    # Create the TTAlchemist instance
    alchemist = TTAlchemist()

    # Create the run config
    config = {"input_file": input or "", "output_file": output or ""}

    # Run the solution
    click.echo(f"Running {model_dir}...")
    if not alchemist.run_solution(model_dir, config):
        click.echo(f"Error: {alchemist.get_last_error()}", err=True)
        sys.exit(1)

    click.echo(f"Solution ran successfully")


@main.command("profile")
@click.argument("model_dir", type=click.Path(exists=True))
@click.option("--input", "-i", help="Input file")
@click.option("--report", "-r", required=True, help="Report file")
def profile(model_dir, input, report):
    """Profile a built solution"""
    # Create the TTAlchemist instance
    alchemist = TTAlchemist()

    # Create the run config
    config = {"input_file": input or "", "output_file": ""}

    # Profile the solution
    click.echo(f"Profiling {model_dir}...")
    if not alchemist.profile_solution(model_dir, config, report):
        click.echo(f"Error: {alchemist.get_last_error()}", err=True)
        sys.exit(1)

    click.echo(f"Solution profiled successfully, report saved to {report}")


@main.command("list-targets")
def list_targets():
    """List available hardware targets"""
    click.echo("Available hardware targets:")
    click.echo("  grayskull")
    click.echo("  wormhole")
    click.echo("  blackhole")


@main.command("list-flavors")
def list_flavors():
    """List available build flavors"""
    click.echo("Available build flavors:")
    click.echo("  release - Optimized for performance")
    click.echo("  debug   - Includes debug symbols")
    click.echo("  profile - Includes profiling instrumentation")


if __name__ == "__main__":
    main()
