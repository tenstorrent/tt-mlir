# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# import _ttmlir_runtime as runtime
import golden


# def golden.py????


def pre_op_callback():
    pass


def post_op_callback():
    print("POST OP CALLBACK CALLED")


def register():
    print("REGISTER CALLED")
    # runtime.register_post_callback(post_op_callback)
