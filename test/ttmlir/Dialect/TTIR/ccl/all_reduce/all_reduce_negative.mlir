// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Unit tests for ttir all_reduce op

module attributes {} {
  func.func @all_reduce_invalid_reduce_type_mean(%arg0: tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32> {
    %0 = ttir.empty() : tensor<1x1x256x256xf32>
    %1 = "ttir.all_reduce"(%arg0, %1) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<mean>}> : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
    return %1 : tensor<1x1x256x256xf32>
  }
}
// CHECK: error: 'ttir.all_reduce' op Invalid reduction op for all reduce op

// -----

module attributes {} {
  func.func @all_reduce_invalid_reduce_type_std(%arg0: tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32> {
    %0 = ttir.empty() : tensor<1x1x256x256xf32>
    %1 = "ttir.all_reduce"(%arg0, %1) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<std>}> : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
    return %1 : tensor<1x1x256x256xf32>
  }
}
// CHECK: error: 'ttir.all_reduce' op Invalid reduction op for all reduce op

// -----

module attributes {} {
  func.func @all_reduce_invalid_reduce_type_var(%arg0: tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32> {
    %0 = ttir.empty() : tensor<1x1x256x256xf32>
    %1 = "ttir.all_reduce"(%arg0, %1) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<var>}> : (tensor<1x1x256x256xf32>, tensor<1x1x256x256xf32>) -> tensor<1x1x256x256xf32>
    return %1 : tensor<1x1x256x256xf32>
  }
}
// CHECK: error: 'ttir.all_reduce' op Invalid reduction op for all reduce op
