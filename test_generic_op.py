import sys
import os

from ttmlir.ir import *
from ttmlir.dialects import func, scf, arith
from ttmlir.dialects import _d2m_ops_gen as d2m

def test_generic():
    with Context() as ctx:
        ctx.allow_unregistered_dialects = True
        with Location.unknown(ctx):
            module = Module.create()
            with InsertionPoint(module.body):
                f32 = F32Type.get()
                tile_type = Type.parse("!ttcore.tile<32x32, f32>", context=ctx)
                layout_attr = None #Attribute.parse(f"#ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, undef, l1, sharded, index_map = map(0)>", context=ctx)
                t_type = RankedTensorType.get([1, 1, 2, 2], tile_type)
                
                func_type = FunctionType.get([t_type, t_type], [t_type])
                func_op = func.FuncOp("test", func_type)
                func_bb = func_op.add_entry_block()
                
                arg0 = func_bb.arguments[0]
                arg1 = func_bb.arguments[1]
                
                with InsertionPoint(func_bb):
                    grid_attr = Attribute.parse("#ttcore.grid<1x1>", context=ctx)
                    block_factors = ArrayAttr.get([])
                    indexing_maps = ArrayAttr.get([])
                    iterator_types = ArrayAttr.get([])
                    threads = ArrayAttr.get([Attribute.parse("#d2m.thread<unified>", context=ctx)])
                    
                    generic_op = d2m.GenericOp(
                        results_=[t_type],
                        inputs=[arg0],
                        outputs=[arg1],
                        grid=grid_attr,
                        block_factors=block_factors,
                        indexing_maps=indexing_maps,
                        iterator_types=iterator_types,
                        threads=threads,
                        num_regions=1
                    )
                    
                    region = generic_op.regions[0]
                    block = region.blocks.append(arg0.type, arg1.type)
                    with InsertionPoint(block):
                        # alloc
                        scratch = d2m.ScratchAllocateOp(RankedTensorType.get([1, 1, 2, 2], tile_type), slot=IntegerAttr.get(IntegerType.get_signless(32), 0))
                        # d2m.yield
                        d2m.YieldOp([block.arguments[1]])
                    
                    func.ReturnOp([generic_op.result])
            print(module)

if __name__ == "__main__":
    test_generic()
