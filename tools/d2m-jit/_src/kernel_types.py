# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


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
