// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-d2m="ttnn-mode=true" --d2m-grid-selection="ttnn-mode=true" --canonicalize %s | FileCheck %s

#l1 = #ttnn.buffer_type<l1>
#input_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x1>, memref<16x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>
#output_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <height_sharded>, exactGrid = true>
module {
  func.func @test_redundant_to_layout_canonicalized(
    %arg0: tensor<4096x32xbf16, #input_layout>) -> tensor<4096x32xbf16, #output_layout> {
    // CHECK-LABEL: func.func @test_redundant_to_layout_canonicalized
    // CHECK: d2m.generic {{{.*}}grid = #ttcore.grid<64x1,
    // CHECK-NOT: d2m.to_layout
    // CHECK: return
    %1 = "ttir.abs"(%arg0)  : (tensor<4096x32xbf16, #input_layout>) -> (tensor<4096x32xbf16>)
    %2 = ttir.empty() : tensor<4096x32xbf16, #output_layout>
    %3 = ttir.to_layout %1, %2 : tensor<4096x32xbf16> into tensor<4096x32xbf16, #output_layout> -> tensor<4096x32xbf16, #output_layout>
    return %3 : tensor<4096x32xbf16, #output_layout>
  }
}
