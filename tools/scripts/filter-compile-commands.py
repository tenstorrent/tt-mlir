# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import sys
import json

assert len(sys.argv) == 3, 'Usage: {} <compile_commands.json> <prefix>'.format(sys.argv[0])

with open(sys.argv[1], 'r') as f:
    compile_commands = json.load(f)

filtered_commands = []
for command in compile_commands:
    if command['file'].startswith(sys.argv[2]):
        filtered_commands.append(command)

with open(sys.argv[1], 'w') as f:
    json.dump(filtered_commands, f, indent=2)
