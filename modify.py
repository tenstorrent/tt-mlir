# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import re
import sys


def main(in_file, out_file):
    loc_pattern = re.compile(r"\sloc\([^)]+\)")
    layout_pattern = re.compile(r",\s#ttnn_layout\*?>")
    with open(in_file, "r") as f:
        content = f.read()

    # Replace , #ttnn_layout*> subsequences with an empty string
    updated_content = layout_pattern.sub("", content)
    # Replace loc(...) subsequences with an empty string
    updated_content = loc_pattern.sub("", updated_content)

    with open(out_file, "w") as f:
        f.write(updated_content)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: remove_loc.py <input_file> <output_file>")
        sys.exit(1)
    main(sys.argv[1], sys.argv[2])
