# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import sys
import re


def convert_type(t):
    if t == "bf16":
        return "bfloat16"
    if t == "f32":
        return "float32"
    return t


def parse_spec(line):
    m = re.search(r"tensor<([^>]+)>", line)
    if not m:
        return None
    raw = m.group(1)
    m2 = re.fullmatch(r"([0-9]+(?:x[0-9]+)*)([a-zA-Z0-9]+)", raw)
    if not m2:
        return None
    shape_str, type_str = m2.groups()
    shape = tuple(map(int, shape_str.split("x")))
    return shape, convert_type(type_str[1:])


def main():
    if len(sys.argv) < 2:
        print("Usage: python parse_mobilenetv3_small.py <file>")
        sys.exit(1)
    with open(sys.argv[1], "r") as f:
        for line in f:
            spec = parse_spec(line)
            if spec:
                print(spec)


if __name__ == "__main__":
    main()
