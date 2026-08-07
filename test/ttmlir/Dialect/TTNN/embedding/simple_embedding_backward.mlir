// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t
module attributes {} {
  func.func @backward(%arg0: tensor<1x32xbf16>, %arg1: tensor<512x128xbf16>, %arg2: tensor<1x32x128xbf16>) -> tensor<512x128xbf16> {
    // Capture reshape output layout for validation
    // CHECK: [[RESHAPE_OUTPUT_LAYOUT:.*]] = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // Verify inserted reshape op
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: <{shape = [1 : i32, 1 : i32, 32 : i32, 128 : i32]}>
    // CHECK-SAME: -> tensor<1x1x32x128xbf16, [[RESHAPE_OUTPUT_LAYOUT]]>
    %1 = "ttir.embedding_backward"(%arg0, %arg1, %arg2) : (tensor<1x32xbf16>, tensor<512x128xbf16>, tensor<1x32x128xbf16>) -> tensor<512x128xbf16>
    // tt-metal returns the weight gradient as (1, 1, dictionary_size, embedding_size),
    // which is reshaped back to the 2D weight shape.
    // CHECK: "ttnn.embedding_bw"
    // CHECK-SAME: -> tensor<1x1x512x128xbf16
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: <{shape = [512 : i32, 128 : i32]}>
    // CHECK-SAME: -> tensor<512x128xbf16
    return %1 : tensor<512x128xbf16>
  }
}
