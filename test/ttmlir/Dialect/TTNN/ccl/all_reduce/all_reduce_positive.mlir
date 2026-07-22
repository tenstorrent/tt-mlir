// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2" -o %t %s
// RUN: FileCheck %s --input-file=%t
// Unit tests for ttnn all_reduce op

// -----

// Verify lowering of ttir all_reduce to a native ttnn all_reduce. A sub-4D
// input is reshaped up to 4D by the reshape workaround and back afterwards.
module attributes {} {
  // CHECK-LABEL: all_reduce_positive_with_reshapes
  func.func @all_reduce_positive_with_reshapes(%arg0: tensor<4096x16384xf32>) -> tensor<4096x16384xf32> {
    %1 = "ttir.all_reduce"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    // CHECK: = "ttnn.reshape"
    // CHECK: "ttnn.all_reduce"
    // CHECK-NOT: "ttnn.reduce_scatter"
    // CHECK-NOT: "ttnn.all_gather"
    // CHECK: = "ttnn.reshape"
    return %1 : tensor<4096x16384xf32>
  }
}

// -----

// Verify lowering of ttir all_reduce to ttnn ops without reshapes for a 4D
// input.
module attributes {} {
  // CHECK-LABEL: all_reduce_positive_without_reshapes
  func.func @all_reduce_positive_without_reshapes(%arg0: tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32> {
    %1 = "ttir.all_reduce"(%arg0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32>
    // CHECK-NOT: = "ttnn.reshape"
    // CHECK: "ttnn.all_reduce"
    // CHECK-NOT: = "ttnn.reshape"
    return %1 : tensor<1x1x4096x16384xf32>
  }
}

// -----

// Verify op folding for single mesh device communication.
module attributes {} {
  // CHECK-LABEL: all_reduce_positive_with_reshapes_folding
  func.func @all_reduce_positive_with_reshapes_folding(%arg0: tensor<4096x16384xf32>) -> tensor<4096x16384xf32> {
    %1 = "ttir.all_reduce"(%arg0) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<4096x16384xf32>) -> tensor<4096x16384xf32>
    // CHECK-NOT: = "ttnn.reshape"
    // CHECK-NOT: "ttnn.all_reduce"
    return %1 : tensor<4096x16384xf32>
  }
}

// -----

// Verify op folding for single mesh device communication.
module attributes {} {
  // CHECK-LABEL: all_reduce_positive_without_reshapes_folding
  func.func @all_reduce_positive_without_reshapes_folding(%arg0: tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32> {
    %1 = "ttir.all_reduce"(%arg0) <{cluster_axis = 0 : ui32, reduce_type = #ttcore.reduce_type<sum>}> : (tensor<1x1x4096x16384xf32>) -> tensor<1x1x4096x16384xf32>
    // CHECK-NOT: = "ttnn.reshape"
    // CHECK-NOT: "ttnn.all_reduce"
    return %1 : tensor<1x1x4096x16384xf32>
  }
}
