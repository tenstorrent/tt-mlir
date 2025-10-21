# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

import os
from enum import Enum

from ttmlir.dialects import ttkernel


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

        return os.path.abspath(file_path)


class CompileTimeValue:
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


class TensorAccessorConfig(Enum):
    """
    Python equivalent ArgConfig from tt_metal/hostdevcommon/api/hostdevcommon/tensor_accessor/arg_config.hpp.
    Bit encoding of fundamental configuration of a tensor accessor that must be available at compile time.
    #TODO: Remove once pybinds for TensorAccessorArgs implemented - Metal Issue #25655
    """

    NONE = 0
    Sharded = 1 << 0  # 0x01 = 0b0000_0001 = 1
    IsDram = 1 << 1  # 0x02 = 0b0000_0010 = 2
    RuntimeRank = 1 << 2  # 0x04 = 0b0000_0100 = 4
    RuntimeNumBanks = 1 << 3  # 0x08 = 0b0000_1000 = 8
    RuntimeTensorShape = 1 << 4  # 0x10 = 0b0001_0000 = 16
    RuntimeShardShape = 1 << 5  # 0x20 = 0b0010_0000 = 32
    RuntimeBankCoords = 1 << 6  # 0x40 = 0b0100_0000 = 64

    def __or__(self, other):
        """Enable | operator"""
        self_val = self.value if hasattr(self, "value") else self
        other_val = other.value if hasattr(other, "value") else other
        return self_val | other_val

    def __ror__(self, other):
        """Enable | operator"""
        return other | self.value

    def __ior__(self, other):
        """Enable |= operator"""
        return self.__or__(other)


class ClassRegistry:
    _registry = {}

    @classmethod
    def register(cls, mlir_type):
        def decorator(attr_class):
            cls._registry[mlir_type] = attr_class
            return attr_class

        return decorator

    @classmethod
    def get(cls, mlir_type):
        return cls._registry.get(mlir_type)

    @classmethod
    def exists(cls, mlir_type):
        return mlir_type in cls._registry


class PyKernelClassBase:
    def __init__(self):
        self.member_functions = {}
        self.member_variables = {}

    def _emit_member_function_mlir(self, member_function, args):
        raise NotImplementedError("emit_mlir must be implemented in subclass")

    def _emit_member_variable_mlir(self, member_variable):
        raise NotImplementedError("emit_mlir must be implemented in subclass")

    def emit_mlir(self, member, args):
        if member in self.member_functions:
            return self._emit_member_function_mlir(member, args)
        elif member in self.member_variables:
            return self._emit_member_variable_mlir(member)
        else:
            raise ValueError(f"Member {member} not found in {self.__class__.__name__}")


@ClassRegistry.register("ttkernel.TensorAccessor")
class TensorAccessor(PyKernelClassBase):
    def __init__(self):
        self.member_functions = {
            "get_noc_addr": ttkernel.tensor_accessor_get_noc_addr,
            "get_shard_noc_addr": ttkernel.tensor_accessor_get_shard_noc_addr,
            "get_bank_and_offset": ttkernel.tensor_accessor_get_bank_and_offset,
            "is_local_bank": ttkernel.tensor_accessor_is_local_bank,
            "is_local_addr": ttkernel.tensor_accessor_is_local_addr,
            "is_local_page": ttkernel.tensor_accessor_is_local_page,
            "is_local_shard": ttkernel.tensor_accessor_is_local_shard,
        }
        self.member_variables = {}

    def _emit_member_function_mlir(self, member_function, args):
        func = self.member_functions[member_function]
        return func(*args)

    def _emit_member_variable_mlir(self, member_variable):
        raise NotImplementedError(
            "TensorAccessorAttributes does not have member variables"
        )
