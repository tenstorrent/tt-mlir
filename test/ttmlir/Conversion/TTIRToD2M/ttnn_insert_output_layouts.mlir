// RUN: ttmlir-opt --convert-ttnn-to-ttir --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" --d2m-grid-selection --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>


// Input Layouts - Paired with Output Layouts
// 1024x1024 DRAM Interleaved
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>, exactGrid = true>
// 2048x2048 on 8x8 L1 Block Sharded
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<8x8x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, exactGrid = true>
// 512x256 on 8x8 L1 Block Sharded
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, exactGrid = true>


// CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 32x32, collapsed_intervals
// CHECK: #layout3 = #ttcore.metal_layout<logical_shape = 2048x2048, dim_alignments = 32x32, collapsed_intervals
// CHECK: #layout5 = #ttcore.metal_layout<logical_shape = 512x256, dim_alignments = 32x32, collapsed_intervals


// Output Layouts
// 1024x1024 on 8x8 L1 Block Sharded
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<4x4x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, exactGrid = true>
// 2048x2048 on 4x4 L1 Block Sharded
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <4x4>, memref<16x16x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, exactGrid = true>
// 512x256 on 8x1 L1 Height Sharded
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x1>, memref<2x8x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>


module {
    // CHECK-LABEL:func.func @dram_to_l1_block_sharded
func.func @dram_to_l1_block_sharded(
    %arg0: tensor<1024x1024xbf16, #ttnn_layout>
) -> tensor<1024x1024xbf16, #ttnn_layout1> {

    //CHECK-NOT: ttir.empty()
    //CHECK-NOT: ttir.to_layout
    %0 = "ttir.abs"(%arg0) : (tensor<1024x1024xbf16, #ttnn_layout>) -> tensor<1024x1024xbf16, #ttnn_layout>

    // CHECK: %[[EMPTY:.*]] = d2m.empty() : tensor<1024x1024xbf16, #ttnn_layout1>
    // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY]] : tensor<1024x1024xbf16, #ttnn_layout1> -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1>
    %1 = ttir.empty() : tensor<1024x1024xbf16, #ttnn_layout1>

    // CHECK: %[[TOLAYOUT:.*]] = d2m.to_layout %3, %[[CAST1]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout2> into tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1>
    // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[TOLAYOUT]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<1024x1024xbf16, #ttnn_layout1>
    %2 = ttir.to_layout %0, %1 : tensor<1024x1024xbf16, #ttnn_layout> into tensor<1024x1024xbf16, #ttnn_layout1> -> tensor<1024x1024xbf16, #ttnn_layout1>

    //CHECK: return %[[CAST2]] : tensor<1024x1024xbf16, #ttnn_layout1>
    return %2 : tensor<1024x1024xbf16, #ttnn_layout1>
    }

    // CHECK-LABEL:func.func @l1_block_sharded_reshard
func.func @l1_block_sharded_reshard(
    %arg0: tensor<2048x2048xbf16, #ttnn_layout2>
) -> tensor<2048x2048xbf16, #ttnn_layout3> {

    //CHECK-NOT: ttir.empty()
    //CHECK-NOT: ttir.to_layout
    %0 = "ttir.abs"(%arg0) : (tensor<2048x2048xbf16, #ttnn_layout2>) -> tensor<2048x2048xbf16, #ttnn_layout2>

    // CHECK: %[[EMPTY:.*]] = d2m.empty() : tensor<2048x2048xbf16, #ttnn_layout3>
    // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY]] : tensor<2048x2048xbf16, #ttnn_layout3> -> tensor<4x4x16x16x!ttcore.tile<32x32, bf16>, #layout3>
    %1 = ttir.empty() : tensor<2048x2048xbf16, #ttnn_layout3>

    // CHECK: %[[TOLAYOUT:.*]] = d2m.to_layout %1, %[[CAST1]] : tensor<8x8x8x8x!ttcore.tile<32x32, bf16>, #layout3> into tensor<4x4x16x16x!ttcore.tile<32x32, bf16>, #layout3> -> tensor<4x4x16x16x!ttcore.tile<32x32, bf16>, #layout3>
    // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[TOLAYOUT]] : tensor<4x4x16x16x!ttcore.tile<32x32, bf16>, #layout3> -> tensor<2048x2048xbf16, #ttnn_layout3>
    %2 = ttir.to_layout %0, %1 : tensor<2048x2048xbf16, #ttnn_layout2> into tensor<2048x2048xbf16, #ttnn_layout3> -> tensor<2048x2048xbf16, #ttnn_layout3>

    //CHECK: return %[[CAST2]] : tensor<2048x2048xbf16, #ttnn_layout3>
    return %2 : tensor<2048x2048xbf16, #ttnn_layout3>
    }

    // CHECK-LABEL:func.func @l1_block_sharded_to_height_sharded
func.func @l1_block_sharded_to_height_sharded(
    %arg0: tensor<512x256xbf16, #ttnn_layout4>
) -> tensor<512x256xbf16, #ttnn_layout5> {

    //CHECK-NOT: ttir.empty()
    //CHECK-NOT: ttir.to_layout
    %0 = "ttir.abs"(%arg0) : (tensor<512x256xbf16, #ttnn_layout4>) -> tensor<512x256xbf16, #ttnn_layout4>

    // CHECK: %[[EMPTY:.*]] = d2m.empty() : tensor<512x256xbf16, #ttnn_layout5>
    // CHECK: %[[CAST1:.*]] = ttir.ttnn_metal_layout_cast %[[EMPTY]] : tensor<512x256xbf16, #ttnn_layout5> -> tensor<8x1x2x8x!ttcore.tile<32x32, bf16>, #layout5>
    %1 = ttir.empty() : tensor<512x256xbf16, #ttnn_layout5>

    // CHECK: %[[TOLAYOUT:.*]] = d2m.to_layout %1, %[[CAST1]] : tensor<8x8x2x1x!ttcore.tile<32x32, bf16>, #layout4> into tensor<8x1x2x8x!ttcore.tile<32x32, bf16>, #layout5> -> tensor<8x1x2x8x!ttcore.tile<32x32, bf16>, #layout5>
    // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %[[TOLAYOUT]] : tensor<8x1x2x8x!ttcore.tile<32x32, bf16>, #layout5> -> tensor<512x256xbf16, #ttnn_layout5>
    %2 = ttir.to_layout %0, %1 : tensor<512x256xbf16, #ttnn_layout4> into tensor<512x256xbf16, #ttnn_layout5> -> tensor<512x256xbf16, #ttnn_layout5>

    //CHECK: return %[[CAST2]] : tensor<512x256xbf16, #ttnn_layout5>
    return %2 : tensor<512x256xbf16, #ttnn_layout5>
    }
}
