# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.dialects import ttkernel


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
