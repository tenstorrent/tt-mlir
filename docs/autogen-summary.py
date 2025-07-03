# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re

os.chdir("src")
autogen_dialect_mds = glob.glob("autogen/*/Dialect/*.md", recursive=True)
autogen_module_mds = glob.glob("autogen/*/Module/*.md", recursive=True)
match_d = re.compile(f"(?P<dialect>\w+)(?P<kind>(Dialect|Op|Type|Attr)).md").match
match_mod = re.compile(r"(?P<module>\w+)\.md").match

d = {}
mod = {}

for md in autogen_dialect_mds:
    m = match_d(os.path.basename(md))
    if m is None:
        continue
    dialect = m.group("dialect")
    if dialect not in d:
        d[dialect] = {}
    d[dialect][m.group("kind")] = (m, md)

for md in autogen_module_mds:
    m = match_mod(os.path.basename(md))
    if m is None:
        continue
    module = m.group("module")
    if module not in mod:
        mod[module] = {}
    mod[module] = md

with open("SUMMARY.md", "a") as fd:
    fd.write("\n---\n")
    fd.write("\n# Dialect Definitions\n")
    for dialect in sorted(d.keys()):
        fd.write(f"- [{dialect}](./{d[dialect]['Dialect'][1]})\n")
        for k in sorted(d[dialect].keys()):
            if k == "Dialect":
                continue
            m, md = d[dialect][k]
            fd.write(f"  - [{dialect}{m.group('kind')}](./{md})\n")

# Read SUMMARY.md into lines
with open("SUMMARY.md", "r") as fd:
    lines = fd.readlines()

# Find the line containing "ttir-builder"
insert_index = None
for i, line in enumerate(lines):
    if "ttir-builder" in line:
        insert_index = i + 1
        break

if insert_index is not None:
    # Prepare module lines (4 spaces indent for mdBook subchapters)
    module_lines = [
        f"    - [{module}](./{mod[module]})\n" for module in sorted(mod.keys())
    ]
    # Insert after the "ttir-builder" line
    lines[insert_index:insert_index] = module_lines

    # Write back the modified lines
    with open("SUMMARY.md", "w") as fd:
        fd.writelines(lines)
else:
    print("Could not find 'ttir-builder' in SUMMARY.md")
