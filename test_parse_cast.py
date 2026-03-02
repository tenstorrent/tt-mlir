from ttmlir.ir import *

with Context() as ctx:
    ctx.allow_unregistered_dialects = True
    module = Module.parse("""
func.func @test(%arg0: tensor<1x1x256x256xf32, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>>>) {
    %0 = "ttir.ttnn_metal_layout_cast"(%arg0) : (tensor<1x1x256x256xf32, #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1x1x8x8x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>>>) -> tensor<1x1x1x1x8x8x!ttcore.tile<32x32, f32>, #ttcore.metal_layout<logical_shape=1x1x1x1x256x256, dim_alignments=1x1x1x1x32x32, collapsed_intervals=dense<> : tensor<0x2xi64>, undef, dram, sharded, index_map = (d0, d1, d2, d3, d4, d5) -> (d0, d1, d2, d3, d4, d5)>>
    return
}
""")
    print("Success!")
