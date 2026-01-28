// RUN: ttmlir-opt --convert-ttnn-to-ttir --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" --d2m-grid-selection --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>

// CHECK: #layout = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 32x32, collapsed_intervals

// Input Layouts
// 1024x1024
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>, exactGrid = true>


// CHECK: #layout1 = #ttcore.metal_layout<logical_shape = 1024x1024, dim_alignments = 32x32, collapsed_intervals

// Output Layouts
// 1024x1024 on 8x8
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<4x4x!ttcore.tile<32x32, bf16>, #l1>, <block_sharded>, exactGrid = true>

module {
    // CHECK-LABEL:func.func @abs
func.func @abs(
    %arg0: tensor<1024x1024xbf16, #ttnn_layout>
) -> tensor<1024x1024xbf16, #ttnn_layout1> {

    // CHECK: %[[EMPTY4:.*]] = d2m.empty() : tensor<1024x1024xbf16, #ttnn_layout1>
    // CHECK: %[[CAST2:.*]] = ttir.ttnn_metal_layout_cast %4 : tensor<1024x1024xbf16, #ttnn_layout1> -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1>
    // CHECK: %[[TOLAYOUT5:.*]] = d2m.to_layout %3, %[[CAST2]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout2> into tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1>
    // CHECK: %[[CAST3:.*]] = ttir.ttnn_metal_layout_cast %[[TOLAYOUT5]] : tensor<8x8x4x4x!ttcore.tile<32x32, bf16>, #layout1> -> tensor<1024x1024xbf16, #ttnn_layout1>

    %0 = "ttir.abs"(%arg0) : (tensor<1024x1024xbf16, #ttnn_layout>) -> tensor<1024x1024xbf16, #ttnn_layout>
    %1 = ttir.empty() : tensor<1024x1024xbf16, #ttnn_layout1>
    %2 = ttir.to_layout %0, %1 : tensor<1024x1024xbf16, #ttnn_layout> into tensor<1024x1024xbf16, #ttnn_layout1> -> tensor<1024x1024xbf16, #ttnn_layout1>
    return %2 : tensor<1024x1024xbf16, #ttnn_layout1>
    }
}
