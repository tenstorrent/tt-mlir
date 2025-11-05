# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# import _ttmlir_runtime as runtime
import golden


def pre_op_callback():
    pass


def post_op_callback(callback_runtime_config, binary, program_context, op_context):
    print("POST OP CALLBACK CALLED")


def post_op_get_callback_fn():
    return partial(post_op_callback)


def register():
    print("REGISTER CALLED")
    callback_env = ttrt.runtime.DebugHooks.get(post_op_get_callback_fn)
    # runtime.register_post_callback(post_op_callback)
