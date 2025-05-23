#!/usr/bin/env python3

# SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
import re
import unicodedata
from pathlib import Path
import argparse
import subprocess
from typing import Dict, Optional
import shutil

# License header text as a docstring
LICENSE_HEADER = """
SPDX-FileCopyrightText: © <YEAR> Tenstorrent AI ULC

SPDX-License-Identifier: Apache-2.0
"""

# Supported file extensions and their comment prefixes
COMMENT_STYLES = {
    ".py": "# ",
    ".sh": "# ",
    ".c": "// ",
    ".cpp": "// ",
    ".cc": "// ",
    ".h": "// ",
    ".hpp": "// ",
    ".cuh": "// ",
    ".cu": "// ",
    ".js": "// ",
    ".ts": "// ",
    ".java": "// ",
    ".go": "// ",
}


# Slow but reliable fallback
def get_git_year(path: Path) -> str:
    try:
        result = subprocess.run(
            [
                "git",
                "log",
                "--diff-filter=A",
                "--follow",
                "--format=%ad",
                "--date=format:%Y",
                "--",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        years = result.stdout.strip().split("\n")
        return years[-1] if years else None
    except subprocess.CalledProcessError:
        return None


def normalize_line(line, normalize_year=True, git_year=None):
    line = unicodedata.normalize("NFKC", line).replace("\ufeff", "").strip()
    if normalize_year:
        # Only normalize if we're not comparing with a specific year
        if git_year is None:
            line = re.sub(r"\b20\d{2}(–20\d{2})?\b", "<YEAR>", line)
    return re.sub(r"\s+", " ", line)


def strip_noise_lines(lines):
    return [
        line for line in lines if line.strip() and not re.match(r"^\s*(//|#)\s*$", line)
    ]


def get_expected_header(file_ext: str, normalize_year=True, git_year=None):
    prefix = COMMENT_STYLES.get(file_ext)
    if not prefix:
        return None

    # If we have a git year and not ignoring years, use that year
    if git_year and not normalize_year:
        header_lines = LICENSE_HEADER.strip().splitlines()
        header_lines = [line.replace("<YEAR>", str(git_year)) for line in header_lines]
        return [normalize_line(f"{prefix}{line}", False) for line in header_lines]

    # Otherwise use the template with <YEAR>
    return [
        normalize_line(f"{prefix}{line}", True)
        for line in LICENSE_HEADER.strip().splitlines()
    ]


def get_raw_expected_header(file_ext: str, git_year=None):
    """Get the raw expected header without normalization for file replacement"""
    prefix = COMMENT_STYLES.get(file_ext)
    if not prefix:
        return None

    header_lines = LICENSE_HEADER.strip().splitlines()

    # If we have a git year, use that year
    if git_year:
        header_lines = [line.replace("<YEAR>", str(git_year)) for line in header_lines]

    return [f"{prefix}{line}" for line in header_lines]


def extract_header_block(path: Path, normalize_year=True, git_year=None):
    try:
        ext = path.suffix
        comment_prefix = COMMENT_STYLES.get(ext)
        if not comment_prefix:
            return None, -1

        lines = []
        header_start_line = -1
        line_num = 0
        with open(path, encoding="utf-8") as f:
            # Read up to the first 15 lines in search of SPDX header block
            content = f.readlines()[:15]  # Read first 15 lines
            for line in content:
                if "SPDX" in line and header_start_line == -1:
                    header_start_line = line_num
                    # Found the start of a SPDX header
                    start_idx = line_num
                    # Get all lines that look like they're part of the header
                    while start_idx < len(content) and (
                        "SPDX" in content[start_idx]
                        or content[start_idx].strip() == comment_prefix.strip()
                    ):
                        lines.append(content[start_idx].rstrip("\n\r"))
                        start_idx += 1
                    break
                line_num += 1

        # Return both the lines and the start line number for replacement purposes
        normalized_lines = [
            normalize_line(line, normalize_year, git_year) for line in lines
        ]
        return normalized_lines, header_start_line
    except Exception as e:
        print(f"❌ ERROR reading {path}: {e}", file=sys.stderr)
        return None, -1


def check_file(
    path: Path,
    expected_lines,
    normalize_year=True,
    git_year=None,
    fix=False,
    only_errors=False,
):
    actual_lines, header_start_line = extract_header_block(
        path, normalize_year, git_year
    )

    # Handle the case where no header was found
    if not actual_lines or len(actual_lines) == 0:
        print(f"❌ No license header found in {path}")
        if fix:
            if add_license_header(path, expected_lines):
                print(f"✅ Added license header to {path}")
                return True
            else:
                print(f"❌ Failed to add license header to {path}")
        return False

    if actual_lines is None:
        return False

    actual = strip_noise_lines(actual_lines)
    expected = strip_noise_lines(expected_lines)

    if actual != expected:
        print(f"❌ Mismatch in {path}")
        print("---- Expected ----")
        print("\n".join(expected))
        print("---- Found ----")
        print("\n".join(actual))
        print()

        if fix and header_start_line >= 0:
            if replace_header(path, expected_lines, header_start_line):
                print(f"✅ Fixed header in {path}")
                return True
            else:
                print(f"❌ Failed to fix header in {path}")
        return False
    elif not only_errors:
        # Only print success messages if not in only_errors mode
        print(f"✅ License header OK in {path}")
    return True


def check_git_requirements():
    # Check if git is installed
    if not shutil.which("git"):
        print("❌ Error: git is not installed or not in PATH", file=sys.stderr)
        return False

    # Check if we're in a git repository
    try:
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        print("❌ Error: Not in a git repository", file=sys.stderr)
        return False

    # Check if repository has history
    try:
        result = subprocess.run(
            ["git", "rev-list", "--count", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        commit_count = int(result.stdout.strip())
        if commit_count == 0:
            print("❌ Error: Git repository has no commit history", file=sys.stderr)
            return False
    except (subprocess.CalledProcessError, ValueError):
        print("❌ Error: Failed to check git history", file=sys.stderr)
        return False

    return True


def get_file_years() -> dict[Path, int | None]:
    try:
        result = subprocess.Popen(
            ["git", "log", "--name-status", "--format=%cs", "--date=short"],
            stdout=subprocess.PIPE,
            universal_newlines=True,
        )
    except subprocess.CalledProcessError as ex:
        print(ex)
        sys.exit(1)

    year: int = 0
    files: dict[str, int | Path] = {}
    for line in result.stdout:
        if (
            not (line)
            or len(line) < 2
            or line[0] == " "
            or line[0] == "M"
            or line[0] == "D"
        ):
            continue

        elif line[0].isdigit():
            year = int(line[0:4])

        elif line[0] == "A":
            files[Path(line.split()[1])] = year

        # this doesn't gracefully handle spaces in paths, so we handle that elsewhere
        elif line[0] == "R":
            split_line = line.split()
            # R<num> <original> <new>
            files[Path(split_line[2])] = Path(split_line[1])
        else:
            continue

    # resolve renames
    for key in files.keys():
        val = files[key]
        while (val is not None) and (type(val) is not int):
            new_val = files.get(val)  # returns None if key doesn't exist
            files[key] = new_val
            val = new_val

    return files


# now you have a master list of all the years for each file


def add_license_header(path: Path, expected_lines):
    """Add a license header to a file that doesn't have one"""
    try:
        ext = path.suffix
        git_year = get_git_year(path)

        # Get the raw expected header (without normalization)
        raw_header = get_raw_expected_header(ext, git_year)
        if not raw_header:
            return False

        # Read the entire file
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Insert header at the beginning of the file
        # For C/C++ files, place license header at the very beginning, before include guards
        if ext in [".h", ".hpp", ".c", ".cpp", ".cc", ".cuh", ".cu"]:
            # For all C/C++ files, license header should come first, then include guards
            new_lines = [line + "\n" for line in raw_header] + ["\n"] + lines
        else:
            # For other files, just insert at the beginning
            new_lines = [line + "\n" for line in raw_header] + ["\n"] + lines

        # Write the file back
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

        return True
    except Exception as e:
        print(f"❌ ERROR adding header to {path}: {e}", file=sys.stderr)
        return False


def replace_header(path: Path, expected_lines, header_start_line):
    """Replace the existing SPDX header with the correct one"""
    try:
        ext = path.suffix
        git_year = get_git_year(path)

        # Get the raw expected header (without normalization)
        raw_header = get_raw_expected_header(ext, git_year)
        if not raw_header:
            return False

        # Read the entire file
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        # Find all SPDX-related lines and empty comment lines in the header area
        header_lines_to_remove = []
        in_header = False
        license_lines = 0
        comment_prefix = COMMENT_STYLES.get(ext, "# ")

        # Scan from the beginning of the file up to 20 lines
        max_scan = min(20, len(lines))
        for i in range(max_scan):
            line = lines[i]
            # If we find a SPDX line, we're in the header
            if "SPDX" in line:
                in_header = True
                header_lines_to_remove.append(i)
                if "SPDX-License-Identifier" in line:
                    license_lines += 1
            # Include empty comment lines and comment-only lines in the header section
            elif in_header and (
                line.strip() == comment_prefix.strip()
                or (
                    line.startswith(comment_prefix)
                    and line.strip() == comment_prefix.strip()
                )
            ):
                header_lines_to_remove.append(i)
            # If we've found the license lines and then hit a non-comment line, we're done with the header
            elif (
                in_header and license_lines > 0 and not line.startswith(comment_prefix)
            ):
                break

        # If we found header lines to remove
        if header_lines_to_remove:
            # Create a new list of lines without the old header
            new_lines = []
            for i in range(len(lines)):
                if i not in header_lines_to_remove:
                    new_lines.append(lines[i])
                elif (
                    i == header_lines_to_remove[0]
                ):  # Insert new header at first removed line
                    for header_line in raw_header:
                        new_lines.append(header_line + "\n")

            # Write the file back
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            return True
        else:
            # Fallback to the original approach if we didn't find any header lines
            current_header_lines = 0
            for i in range(header_start_line, min(header_start_line + 10, len(lines))):
                if "SPDX" in lines[i]:
                    current_header_lines += 1
                    # Look for additional header lines (typically 2 more after the SPDX line)
                    for j in range(1, 3):
                        if (i + j < len(lines)) and (
                            lines[i + j].startswith(comment_prefix)
                            or lines[i + j].strip() == ""
                        ):
                            current_header_lines += 1
                    break

            # Remove the current header and insert the new one
            header_end_line = header_start_line + current_header_lines
            new_lines = (
                lines[:header_start_line]
                + [line + "\n" for line in raw_header]
                + lines[header_end_line:]
            )

            # Write the file back
            with open(path, "w", encoding="utf-8") as f:
                f.writelines(new_lines)

            return True

    except Exception as e:
        print(f"❌ ERROR replacing header in {path}: {e}", file=sys.stderr)
        return False


def main():
    parser = argparse.ArgumentParser(description="Check license headers in files.")
    parser.add_argument(
        "--ignore-year", action="store_true", help="Ignore year differences."
    )
    parser.add_argument(
        "--fix",
        action="store_true",
        help="Replace incorrect SPDX headers with correct ones.",
    )
    parser.add_argument(
        "--only-errors",
        action="store_true",
        help="Only print messages for files with errors.",
    )
    parser.add_argument("files", nargs="+", help="Files to check")
    args = parser.parse_args()

    # Check git requirements before proceeding
    if not check_git_requirements():
        sys.exit(1)

    file_years = get_file_years()

    failed = False
    for file_arg in args.files:
        path = Path(file_arg)
        ext = path.suffix

        # Only print checking message if not in only-errors mode
        if not args.only_errors:
            print(f"Checking {path} with ext {ext}", file=sys.stderr)

        git_year = file_years.get(path)

        # the fast lookup table doesn't handle spaces in file names, so when
        # we come across a file that was named incorrectly, use the slower
        # method as a reliable fallback.
        if git_year is None:
            git_year = get_git_year(path)

        if not args.only_errors:
            print(f"Git year: {git_year}", file=sys.stderr)

        expected = get_expected_header(ext, args.ignore_year, git_year)

        if expected is None:
            continue  # Skip unsupported files

        # Pass the only_errors flag to check_file
        if not check_file(
            path,
            expected,
            args.ignore_year,
            git_year,
            fix=args.fix,
            only_errors=args.only_errors,
        ):
            failed = True

    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
