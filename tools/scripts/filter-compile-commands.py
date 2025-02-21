# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import re
import subprocess
import sys


def get_diff_files(compile_commands):
    fork_point = (
        subprocess.check_output(["git", "merge-base", "--fork-point", "main"])
        .decode("utf-8")
        .strip()
    )
    diff = subprocess.check_output(["git", "diff", "--name-only", fork_point]).decode(
        "utf-8"
    )
    cwd = os.getcwd()
    processed = map(lambda x: os.path.join(cwd, x.strip()), diff.split("\n"))
    return set(
        filter(
            lambda x: x.endswith(".c")
            or x.endswith(".cc")
            or x.endswith(".cpp")
            or x.endswith(".hpp")
            or x.endswith(".h"),
            processed,
        )
    )


def get_deps(command):
    try:
        cmd = command["command"].split(" ")
        try:
            idx = cmd.index("-o")
            del cmd[idx : idx + 2]
        except:
            pass
        cmd.insert(1, "-MM")
        deps = subprocess.check_output(cmd, cwd=command["directory"]).decode("utf-8")
        return set(map(lambda x: x.strip(" \\"), deps.split("\n")[1:]))
    except:
        print("Failed to get deps for", command["file"])
        print("Command:", command["command"])
        return set()


def main(args):
    with open(args.compile_commands, "r") as f:
        compile_commands = json.load(f)

    filtered_commands = []
    m = re.compile(r"^{}/((?!third_party).)*$".format(args.prefix))
    diff_files = get_diff_files(compile_commands) if args.diff else set()
    compile_commands = sorted(
        compile_commands, key=lambda x: x["file"] not in diff_files
    )
    for command in compile_commands:
        if args.diff and not diff_files:
            break
        if not m.match(command["file"]):
            continue
        if args.diff and command["file"] not in diff_files:
            deps = get_deps(command) & diff_files
            if not deps:
                continue
            diff_files = diff_files - deps
        filtered_commands.append(command)
        diff_files = diff_files - {command["file"]}

    if args.dry:
        json.dump(filtered_commands, sys.stdout, indent=2)
        sys.stdout.write("\n")
    else:
        with open(args.compile_commands, "w") as f:
            json.dump(filtered_commands, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Filter compile commands to exclude unnecessary files for linting."
    )
    parser.add_argument("compile_commands", help="Path to compile_commands.json")
    parser.add_argument(
        "--prefix",
        default=os.getcwd(),
        help="Prefix to filter out from the compile commands.",
    )
    parser.add_argument(
        "--diff", action="store_true", help="Filter out files that are not in the diff."
    )
    parser.add_argument(
        "--dry",
        action="store_true",
        help="Dry run, do not write to compile_commands.json.",
    )

    args = parser.parse_args()
    main(args)
