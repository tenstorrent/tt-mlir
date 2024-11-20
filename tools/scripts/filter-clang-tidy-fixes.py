# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import yaml
import sys

with open(sys.argv[1], "r") as fd:
    d = yaml.load(fd, yaml.SafeLoader)

fixes = d["Diagnostics"]
uniques = set([str(f) for f in fixes])
unique_fixes = []
for f in fixes:
    if str(f) in uniques:
        unique_fixes.append(f)
        uniques.remove(str(f))
d["Diagnostics"] = unique_fixes

with open(sys.argv[1], "w") as fd:
    yaml.dump(d, fd)
