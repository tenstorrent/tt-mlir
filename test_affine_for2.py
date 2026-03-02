import sys
from ttmlir.ir import *
from ttmlir.dialects import func, affine, arith

with Context() as ctx, Location.unknown(ctx):
    ctx.allow_unregistered_dialects = True
    module = Module.create()
    with InsertionPoint(module.body):
        loop = affine.AffineForOp(0, 128, 32)
        with InsertionPoint(loop.body):
            affine.AffineYieldOp([])
        print(module)
