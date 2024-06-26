# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import glob
import os
import re

os.chdir("src")
autogen_mds = glob.glob("autogen/**/*.md", recursive=True)
match = re.compile(f"(?P<dialect>\w+)(?P<kind>(Dialect|Op|Type|Attr)).md").match
d = {}

for md in autogen_mds:
    m = match(os.path.basename(md))
    if m is None:
        continue
    dialect = m.group("dialect")
    if dialect not in d:
        d[dialect] = {}
    d[dialect][m.group("kind")] = (m, md)

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
