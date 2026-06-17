#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

"""CLI tool for managing pattern test YAML configurations.

Usage:
    python yaml_cli.py generate <pattern_name> <pattern_module>
    python yaml_cli.py validate [<config_file>]
    python yaml_cli.py list
    python yaml_cli.py migrate <pattern_file>
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from pattern_tests.yaml_loader import (
    generate_yaml_template,
    load_yaml_config,
    discover_yaml_configs,
)
from pattern_tests.config_schema import validate_config
from pattern_tests.discovery import get_patterns_dir


def cmd_generate(args):
    """Generate a YAML template for a new pattern."""
    template = generate_yaml_template(args.pattern_name, args.pattern_module)

    if args.output:
        output_file = Path(args.output)
    else:
        patterns_dir = get_patterns_dir()
        output_file = patterns_dir / f"{args.pattern_module}.test.yaml"

    if output_file.exists() and not args.force:
        print(f"Error: {output_file} already exists. Use --force to overwrite.")
        return 1

    output_file.write_text(template)
    print(f"Generated template: {output_file}")
    print("\nEdit the file to add your test cases, then run:")
    print(f"  python yaml_cli.py validate {output_file}")
    return 0


def cmd_validate(args):
    """Validate YAML configuration file(s)."""
    if args.config_file:
        config_files = [Path(args.config_file)]
    else:
        # Validate all YAML configs
        patterns_dir = get_patterns_dir()
        config_files = list(patterns_dir.glob("*.test.yaml"))

    if not config_files:
        print("No YAML config files found.")
        return 1

    all_valid = True
    for config_file in config_files:
        print(f"\nValidating {config_file.name}...")

        try:
            config = load_yaml_config(config_file)
            if config is None:
                print(f"  ⨯ File is empty or invalid")
                all_valid = False
                continue

            errors = validate_config(config)
            if errors:
                print(f"  ⨯ Validation failed:")
                for error in errors:
                    print(f"      - {error}")
                all_valid = False
            else:
                print(f"  ✓ Valid")
                print(f"      Pattern: {config.pattern_name}")
                print(f"      LIT tests: {len(config.lit_tests)}")
                print(f"      E2E tests: {len(config.e2e_tests)}")

        except Exception as e:
            print(f"  ⨯ Error: {e}")
            all_valid = False

    return 0 if all_valid else 1


def cmd_list(args):
    """List all YAML configurations."""
    patterns_dir = get_patterns_dir()
    configs = discover_yaml_configs(patterns_dir, load_modules=False)

    if not configs:
        print("No YAML configurations found.")
        return 0

    print(f"\nFound {len(configs)} pattern configuration(s):\n")

    for config in configs:
        print(f"  {config.pattern_name}")
        print(f"    Module: {config.pattern_module}")
        print(
            f"    File: {config._config_file.name if config._config_file else 'unknown'}"
        )
        print(f"    Tests: {len(config.lit_tests)} LIT, {len(config.e2e_tests)} E2E")
        if config.tags:
            print(f"    Tags: {', '.join(config.tags)}")
        print()

    return 0


def cmd_migrate(args):
    """Migrate legacy dict-based config to YAML (WIP)."""
    print("Migration from dict to YAML is not yet implemented.")
    print("Please manually create YAML config using:")
    print(f"  python yaml_cli.py generate <pattern_name> <pattern_module>")
    return 1


def main():
    parser = argparse.ArgumentParser(
        description="Manage pattern test YAML configurations"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate YAML template")
    generate_parser.add_argument(
        "pattern_name", help="Pattern name (e.g., eltwise_exp)"
    )
    generate_parser.add_argument(
        "pattern_module", help="Pattern module name (e.g., eltwise_exp_to_kernel)"
    )
    generate_parser.add_argument("-o", "--output", help="Output file path")
    generate_parser.add_argument(
        "-f", "--force", action="store_true", help="Overwrite existing file"
    )

    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate YAML config(s)")
    validate_parser.add_argument(
        "config_file", nargs="?", help="Config file to validate (default: all)"
    )

    # List command
    list_parser = subparsers.add_parser("list", help="List all YAML configs")

    # Migrate command
    migrate_parser = subparsers.add_parser(
        "migrate", help="Migrate dict config to YAML"
    )
    migrate_parser.add_argument(
        "pattern_file", help="Pattern .py file with PATTERN_TEST_METADATA"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    commands = {
        "generate": cmd_generate,
        "validate": cmd_validate,
        "list": cmd_list,
        "migrate": cmd_migrate,
    }

    return commands[args.command](args)


if __name__ == "__main__":
    sys.exit(main())
