# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os


class CircularBuffer:
    def __init__(self, cb_id, tensor_shape=(8, 128, 128), dtype="Float32"):
        self.cb_id = cb_id
        self.tensor_shape = tensor_shape
        self.tile_shape = 32  # default to 32x32 tile shape
        self.tilized_shape = self.get_tilized_memref_shape()
        self.dtype = dtype

    def get_tilized_memref_shape(self):
        tilized_shape = list(self.tensor_shape)
        tilized_shape[-2] = (tilized_shape[-2] + self.tile_shape - 1) // self.tile_shape
        tilized_shape[-1] = (tilized_shape[-1] + self.tile_shape - 1) // self.tile_shape
        return tilized_shape


class Kernel:
    def __init__(self, kernel_path):
        self.kernel_path = kernel_path
        self.kernel_name = os.path.splitext(os.path.basename(kernel_path))[0]

    def dump(self):
        with open(self.kernel_path, "r") as f:
            print(f.read())
