// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Unit tests for ttir reduce_scatter op

// -----

module attributes {} {
  func.func @reduce_scatter_invalid_reduce_type_mean(%arg0: tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32> {
    %0 = ttir.empty() : tensor<1x1x8192x256xf32>
    %1 = "ttir.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<mean>, scatter_dim = 3 : si32}> : (tensor<1x1x8192x256xf32>, tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    return %1 : tensor<1x1x8192x256xf32>
  }
}
// CHECK: error: 'ttir.reduce_scatter' op Invalid reduction op for reduce scatter op

// -----

module attributes {} {
  func.func @reduce_scatter_invalid_reduce_type_std(%arg0: tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32> {
    %0 = ttir.empty() : tensor<1x1x8192x256xf32>
    %1 = "ttir.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<std>, scatter_dim = 3 : si32}> : (tensor<1x1x8192x256xf32>, tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    return %1 : tensor<1x1x8192x256xf32>
  }
}
// CHECK: error: 'ttir.reduce_scatter' op Invalid reduction op for reduce scatter op

// -----

module attributes {} {
  func.func @reduce_scatter_invalid_reduce_type_var(%arg0: tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32> {
    %0 = ttir.empty() : tensor<1x1x8192x256xf32>
    %1 = "ttir.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<var>, scatter_dim = 3 : si32}> : (tensor<1x1x8192x256xf32>, tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    return %1 : tensor<1x1x8192x256xf32>
  }
}
// CHECK: error: 'ttir.reduce_scatter' op Invalid reduction op for reduce scatter op

// -----

module attributes {} {
  func.func @reduce_scatter_invalid_dim(%arg0: tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32> {
    %0 = ttir.empty() : tensor<1x1x8192x256xf32>
    %1 = "ttir.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = 4 : si32}> : (tensor<1x1x8192x256xf32>, tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    return %1 : tensor<1x1x8192x256xf32>
  }
}
// CHECK: error: 'ttir.reduce_scatter' op Invalid dimension for reduce scatter op. Scatter dimension must be >= to input tensor rank or < -input tensor rank, got scatter_dim = 4

// -----

module attributes {} {
  func.func @reduce_scatter_invalid_negative_dim(%arg0: tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32> {
    %0 = ttir.empty() : tensor<1x1x8192x256xf32>
    %1 = "ttir.reduce_scatter"(%arg0, %0) <{cluster_axis = 1 : ui32, reduce_type = #ttcore.reduce_type<sum>, scatter_dim = -5 : si32}> : (tensor<1x1x8192x256xf32>, tensor<1x1x8192x256xf32>) -> tensor<1x1x8192x256xf32>
    return %1 : tensor<1x1x8192x256xf32>
  }
}
// CHECK: error: 'ttir.reduce_scatter' op Invalid dimension for reduce scatter op. Scatter dimension must be >= to input tensor rank or < -input tensor rank, got scatter_dim = -5
