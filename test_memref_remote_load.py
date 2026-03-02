import sys
from ttmlir.ir import *
from ttmlir.dialects import func, scf, arith, memref
from ttmlir.dialects import _d2m_ops_gen as d2m

with Context() as ctx, Location.unknown(ctx):
    ctx.allow_unregistered_dialects = True
    module = Module.create()
    with InsertionPoint(module.body):
        f32 = F32Type.get()
        tile_type = Type.parse("!ttcore.tile<32x32, f32>", context=ctx)
        l1_space = Attribute.parse("#ttcore.memory_space<l1>", context=ctx)
        dram_space = Attribute.parse("#ttcore.memory_space<dram>", context=ctx)
        
        l1_memref_type = MemRefType.get([1, 1], tile_type, memory_space=l1_space)
        dram_memref_type = MemRefType.get([8, 8], tile_type, memory_space=dram_space)

        alloc = memref.AllocOp(l1_memref_type, [], [])
        
        # Mock up arguments
        dram_arg = arith.ConstantOp(F32Type.get(), 0.0) # Dummy
        
        # d2m_dialect.RemoteLoadOp(result, memref, indices, mcastStartIndex, mcastShape, mcastDims, localBuffer, cb)
        try:
            # Let's see if we can use it with memref
            # Need an op that produces a memref for dram_arg. We can just use an alloc for dram as well.
            dram_alloc = memref.AllocOp(dram_memref_type, [], [])
            
            idx0 = arith.ConstantOp(IndexType.get(), 0)
            idx1 = arith.ConstantOp(IndexType.get(), 0)
            
            mcast_empty = ArrayAttr.get([])
            
            rl = d2m.RemoteLoadOp(
                l1_memref_type, 
                dram_alloc.result, 
                [idx0.result, idx1.result],
                mcast_empty, mcast_empty, mcast_empty,
                localBuffer=alloc.result
            )
            print("RemoteLoadOp:", rl)
            print("Result type:", rl.result.type)
        except Exception as e:
            print("Error:", e)
