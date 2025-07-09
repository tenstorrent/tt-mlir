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
    def __init__(self, kernel_name, kernel_string):
        self.kernel_string = kernel_string
        self.kernel_name = kernel_name

    def dump(self):
        print(self.kernel_string)

    def dump_to_file(self, file_path=""):
        if not file_path:
            file_path = f"generated/pykernels/{self.kernel_name}.cpp"
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "w") as f:
            f.write(self.kernel_string)

        return file_path


class CompiledValue:
    def __init__(self, key, value):
        self.key = key
        self.value = value


class Arguments:
    def __init__(self, x=0, y=0, is_common=False, common_idx=None):
        if is_common is True:
            if common_idx is None:
                raise Exception("If argument is common, Common_idx must be defined")
            self.common_idx = common_idx
            self.is_common = True
        else:
            self.args = [[[] for i in range(x)] for j in range(y)]

    @staticmethod
    def make_common(val, idx):
        res = Arguments(0, 0, is_common=True, common_idx=idx)
        setattr(res, "value", val)
        return res

    def set_args_at_core(self, i, j, args):
        self.args[i][j] = args

    def add_args_at_core(self, i, j, args):
        self.args[i][j].extend(args)

    def get_args(self, x=0, y=0):
        return self.args[x][y]

    def get_all_args(self):
        return self.args
