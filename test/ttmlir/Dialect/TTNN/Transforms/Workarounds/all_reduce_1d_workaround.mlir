// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2" -o %t %s
// RUN: FileCheck %s --input-file=%t
// Unit tests for all_reduce 1D tensor reshape workaround

// Verify that 1D tensor all_reduce is reshaped to 2D, then decomposed into
// reduce_scatter + all_gather (by TTNNAllReduceWorkarounds and
// ReduceScatterOpRewritePattern which further pads to 4D), then reshaped
// back to 1D.

// -----

module attributes {} {
  // CHECK-LABEL: all_reduce_1d_reshape_workaround_128
  func.func @all_reduce_1d_reshape_workaround_128(%arg0: tensor<128xbf16>) -> tensor<128xbf16> {
    %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<128xbf16>) -> tensor<128xbf16>
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.reduce_scatter"
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [128 : i32]
    return %0 : tensor<128xbf16>
  }
}

// -----

module attributes {} {
  // CHECK-LABEL: all_reduce_1d_reshape_workaround_64
  func.func @all_reduce_1d_reshape_workaround_64(%arg0: tensor<64xbf16>) -> tensor<64xbf16> {
    %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<64xbf16>) -> tensor<64xbf16>
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.reduce_scatter"
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [64 : i32]
    return %0 : tensor<64xbf16>
  }
}

// -----

module attributes {} {
  // CHECK-LABEL: all_reduce_1d_reshape_workaround_256
  func.func @all_reduce_1d_reshape_workaround_256(%arg0: tensor<256xbf16>) -> tensor<256xbf16> {
    %0 = "ttir.all_reduce"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<256xbf16>) -> tensor<256xbf16>
    // CHECK: "ttnn.reshape"
    // CHECK: "ttnn.reduce_scatter"
    // CHECK: "ttnn.all_gather"
    // CHECK: "ttnn.reshape"
    // CHECK-SAME: shape = [256 : i32]
    return %0 : tensor<256xbf16>
  }
}
