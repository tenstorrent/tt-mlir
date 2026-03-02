import sys
from ttmlir.ir import *
from ttmlir.dialects import func, scf, arith, memref

with Context() as ctx, Location.unknown(ctx):
    ctx.allow_unregistered_dialects = True
    module = Module.create()
    with InsertionPoint(module.body):
        f32 = F32Type.get()
        tile_type = Type.parse("!ttcore.tile<32x32, f32>", context=ctx)
        
        # Test MemRefType
        l1_space = Attribute.parse("#ttcore.memory_space<l1>", context=ctx)
        l1_memref = MemRefType.get([1, 1], tile_type, memory_space=l1_space)
        print("L1 MemRefType:", l1_memref)
        
        dram_space = Attribute.parse("#ttcore.memory_space<dram>", context=ctx)
        dram_memref = MemRefType.get([8, 8], tile_type, memory_space=dram_space)
        print("DRAM MemRefType:", dram_memref)

        alloc = memref.AllocOp(l1_memref, [], [])
        print("AllocOp:", alloc)
