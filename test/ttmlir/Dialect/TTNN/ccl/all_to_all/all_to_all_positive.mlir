// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-backend-pipeline="mesh-shape=1,2" %s 2>&1 | FileCheck %s
// Unit tests for ttnn all_to_all op

// -----

module attributes {} {
  // CHECK-LABEL: all_to_all_basic
  func.func public @all_to_all_basic(%arg0: tensor<1x1x32x32xf32>) -> (tensor<1x1x32x32xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x1x32x32xf32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{cluster_axis = 1 : ui32, concat_dim = 3 : si32, split_count = 2 : si32, split_dim = 3 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x1x32x32xf32>) -> tensor<1x1x32x32xf32>
    // CHECK: "ttnn.all_to_all"
    // CHECK-SAME: cluster_axis = 1 : ui32
    // CHECK-SAME: concat_dim = 3 : si32
    // CHECK-SAME: split_count = 2 : si32
    // CHECK-SAME: split_dim = 3 : si32
    return %1 : tensor<1x1x32x32xf32>
  }
}

// -----

module attributes {} {
  // CHECK-LABEL: all_to_all_different_dim
  func.func public @all_to_all_different_dim(%arg0: tensor<1x1x32x32xf32>) -> (tensor<1x2x32x16xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<1x2x32x16xf32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{cluster_axis = 1 : ui32, concat_dim = 1 : si32, split_count = 2 : si32, split_dim = 3 : si32}> : (tensor<1x1x32x32xf32>, tensor<1x2x32x16xf32>) -> tensor<1x2x32x16xf32>
    // CHECK: "ttnn.all_to_all"
    // CHECK-SAME: cluster_axis = 1 : ui32
    // CHECK-SAME: concat_dim = 1 : si32
    // CHECK-SAME: split_count = 2 : si32
    // CHECK-SAME: split_dim = 3 : si32
    return %1 : tensor<1x2x32x16xf32>
  }
}

// -----

module attributes {} {
  // CHECK-LABEL: all_to_all_basic_2d
  func.func public @all_to_all_basic_2d(%arg0: tensor<128x128xf32>) -> (tensor<128x128xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<128x128xf32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{cluster_axis = 1 : ui32, concat_dim = 1 : si32, split_count = 2 : si32, split_dim = 1 : si32}> : (tensor<128x128xf32>, tensor<128x128xf32>) -> tensor<128x128xf32>
    // CHECK: "ttnn.all_to_all"
    // CHECK-SAME: cluster_axis = 1 : ui32
    // CHECK-SAME: concat_dim = 1 : si32
    // CHECK-SAME: split_count = 2 : si32
    // CHECK-SAME: split_dim = 1 : si32
    return %1 : tensor<128x128xf32>
  }
}

// -----

module attributes {} {
  // CHECK-LABEL: all_to_all_different_dim_2d
  func.func public @all_to_all_different_dim_2d(%arg0: tensor<128x128xf32>) -> (tensor<256x64xf32> {jax.result_info = ""}) {
    %0 = ttir.empty() : tensor<256x64xf32>
    %1 = "ttir.all_to_all"(%arg0, %0) <{cluster_axis = 1 : ui32, concat_dim = 0 : si32, split_count = 2 : si32, split_dim = 1 : si32}> : (tensor<128x128xf32>, tensor<256x64xf32>) -> tensor<256x64xf32>
    // CHECK: "ttnn.all_to_all"
    // CHECK-SAME: cluster_axis = 1 : ui32
    // CHECK-SAME: concat_dim = 0 : si32
    // CHECK-SAME: split_count = 2 : si32
    // CHECK-SAME: split_dim = 1 : si32
    return %1 : tensor<256x64xf32>
  }
}
