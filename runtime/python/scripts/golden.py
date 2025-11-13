# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import golden
import ttrt.runtime


def pre_op_callback(binary, program_context, op_context):
    print("PRE OP CALLBACK CALLED")


def post_op_callback(binary, program_context, op_context):
    print("POST OP CALLBACK CALLED")


def register(message: str):
    print("REGISTER CALLED")
    print(f"Message from C++ runtime: {message}")

    callback_env = ttrt.runtime.DebugHooks.get(pre_op_callback, post_op_callback)
    print("REGISTER CALLBACK ENVIRONMENT:", callback_env)
