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
        memref_type = MemRefType.get([1, 1], tile_type, memory_space=l1_space)

        alloc1 = memref.AllocOp(memref_type, [], [])
        alloc2 = memref.AllocOp(memref_type, [], [])
        alloc_out = memref.AllocOp(memref_type, [], [])

        identity_map = AffineMap.get_identity(2)
        indexing_maps = ArrayAttr.get([
            AffineMapAttr.get(identity_map),
            AffineMapAttr.get(identity_map),
            AffineMapAttr.get(identity_map),
        ])
        iterator_types = ArrayAttr.get([StringAttr.get("parallel"), StringAttr.get("parallel")])

        # Try to create linalg.generic with memrefs
        linalg_generic = Operation.create(
            "linalg.generic",
            results=[],
            operands=[alloc1.result, alloc2.result, alloc_out.result],
            attributes={
                "indexing_maps": indexing_maps,
                "iterator_types": iterator_types,
                "operandSegmentSizes": DenseI32ArrayAttr.get([2, 1])
            },
            regions=1,
        )
        region = linalg_generic.regions[0]
        block = region.blocks.append(tile_type, tile_type, tile_type)
        with InsertionPoint(block):
            tile_add = d2m.TileAddOp(tile_type, block.arguments[0], block.arguments[1])
            Operation.create("linalg.yield", operands=[tile_add.result])
            
        print(module)
