# SPDX-FileCopyrightText: (c) 2024 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

from ttmlir.ir import *
from ttmlir.dialects import tt, ttir, func, scf, arith
import torch
import traceback


class TensorGolden:
    def __init__(self, value, **kwargs):
        if type(value) is torch.Tensor:
            self.kind = "torch.tensor"
            self.value = storage
        elif type(value) is tuple or type(value) is list:
            self.kind = "seed"
            self.value = value
        else:
            raise ValueError("Invalid value")


class TTIRBuilder:
    def __init__(self):
        self.ctx = Context()
        tt.register_dialect(self.ctx)
        self.module = Module.create(self.cursor)

    def _get_loc(self, depth=2):
        tb = traceback.extract_stack(limit=depth + 1)
        if len(tb) <= depth:
            return Location.unknown(self.ctx)
        col = 0
        return Location.file(tb[depth].filename, tb[depth].lineno, col, context=self.ctx)

    def build_func(self, builder_fn):
        name = builder_fn.__name__
        block = Block.create_at_start(entry.body, operandTypes)
        builder_fn(self)

    def tensor(self, shape, dtype=None, encoding=None):
        if dtype is None:
            dtype = F32Type.get(self.ctx)
        return RankedTensorType.get(shape, dtype, encoding, self._get_loc())


def TupleTensorBuilder:
    def __init__(self, shape):
        self.shape = shape

    def build(self, builder):
        return builder.tensor(self.shape)


def func(inputs=[], outputs=[]):
    def invokeTensorBuilder(builder, tensor):
        if type(tensor) is tuple or type(tensor) is list:
            tensor = TupleTensorBuilder(tensor)

        if hasattr(tensor, "build"):
            return tensor.build(builder)
        else:
            raise ValueError(f"Invalid tensor builder, expected build method: {type(tensor)}")

    builder = TTIRBuilder()
    inputs = [invokeTensorBuilder(builder, i) for i in inputs]
    outputs = [invokeTensorBuilder(builder, i) for i in outputs]
    builder.build_func(test_add)
    def wrapper(fn):
        def wrapped(builder):
            inputs = fn(builder)
            return func(inputs, outputs)

        return wrapped

    return wrapper


@func(inputs=[(32, 32), (32, 32)], outputs=[(32, 32)])
def test_add(builder):
    torch.seed(0)
    a_golden = torch.randn(32, 32)
    torch.seed(1)
    b_golden = torch.randn(32, 32)
    golden = a_golden + b_golden
    a = builder.tensor([32, 32])
    b = builder.tensor([32, 32])
    out = builder.add(a, b)
    return [TensorGolden((32, 32)), TensorGolden((32, 32))], [TensorGolden(golden)]
