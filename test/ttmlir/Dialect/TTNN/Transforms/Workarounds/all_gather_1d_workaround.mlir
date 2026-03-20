// RUN: ttmlir-opt --ttcore-register-device --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t
// Unit tests for all_gather 1D tensor reshape workaround

// Verify that 1D tensor all_gather is decomposed into reshape(2D) -> all_gather -> reshape(1D)

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0) -> (d0, 0), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_out = #ttnn.ttnn_layout<(d0) -> (d0, 0), <1x1>, memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {} {
  // CHECK-LABEL: all_gather_1d_reshape_workaround
  func.func @all_gather_1d_reshape_workaround(%arg0: tensor<32xbf16, #ttnn_layout>) -> tensor<128xbf16, #ttnn_layout_out> {
    // CHECK: %[[RESHAPE_IN:.*]] = "ttnn.reshape"(%arg0)
    // CHECK-SAME: shape = [1 : i32, 32 : i32]
    // CHECK: %[[AG:.*]] = "ttnn.all_gather"(%[[RESHAPE_IN]])
    // CHECK-SAME: all_gather_dim = 1 : si32
    // CHECK: "ttnn.reshape"(%[[AG]])
    // CHECK-SAME: shape = [128 : i32]
    %0 = "ttnn.all_gather"(%arg0) <{all_gather_dim = 0 : si32, cluster_axis = 1 : ui32}> : (tensor<32xbf16, #ttnn_layout>) -> tensor<128xbf16, #ttnn_layout_out>
    return %0 : tensor<128xbf16, #ttnn_layout_out>
  }
}
