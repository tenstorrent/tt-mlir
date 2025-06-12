#!/usr/bin/env python3
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import hashlib
import sys

namespace = sys.argv[1]
variable_name = sys.argv[2]
in_file_name = sys.argv[3]
out_file_name = sys.argv[4]

in_file_contents = open(in_file_name).read().encode()
sha = hashlib.sha256(in_file_contents).hexdigest()
define = f"TTMLIR_SHA256_INCLUDE_GEN_{namespace.replace('::', '_').upper()}_{variable_name.upper()}"

out_file = open(out_file_name, "w")
out_file.write(
    f"""// Auto-generated do not edit directly
#ifndef {define}
#define {define}
namespace {namespace} {{
static constexpr char {variable_name}[] = \"{sha}\";
}}
#endif // {define}
"""
)
