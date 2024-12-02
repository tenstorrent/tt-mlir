// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="enable-optimizer=true" %s | FileCheck %s
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: #[[LAYOUT_2:.*]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<1x1x!tt.tile<32x32, f32>, #dram>, interleaved>
    // CHECK: %[[C:.*]] = "ttnn.empty"[[C:.*]]
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.multiply"[[C:.*]] -> tensor<64x128xf32, #[[LAYOUT_2]]>
    %1 = "ttir.multiply"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }
}
