// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func public @argmax_3d(%arg0: tensor<128x28x28xf32>) -> tensor<28xi32> {
    // CHECK-LABEL: func.func public @argmax_3d
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0, 2>}> : (tensor<128x28x28xf32>) -> tensor<28x128x28xf32>
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[PERMUTE]]) <{shape = [28 : i32, 3584 : i32]}> : (tensor<28x128x28xf32>) -> tensor<28x3584xf32>
    // CHECK: %[[ARGMAX:[0-9]+]] = "ttir.argmax"(%[[RESHAPE]]) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<28x3584xf32>) -> tensor<28xi32>
    // CHECK: return %[[ARGMAX]] : tensor<28xi32>
    %1 = "ttir.argmax"(%arg0) <{dim_arg = [0: i32, 2 : i32], keep_dim = false}> : (tensor<128x28x28xf32>) -> tensor<28xi32>
    return %1 : tensor<28xi32>
  }

  func.func public @argmax_3d_keep(%arg0: tensor<128x28x28xf32>) -> tensor<1x28x1xi32> {
    // CHECK-LABEL: func.func public @argmax_3d_keep
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0, 2>}> : (tensor<128x28x28xf32>) -> tensor<28x128x28xf32>
    // CHECK: %[[FIRST_RESHAPE:[0-9]+]] = "ttir.reshape"(%[[PERMUTE]]) <{shape = [28 : i32, 3584 : i32]}> : (tensor<28x128x28xf32>) -> tensor<28x3584xf32>
    // CHECK: %[[ARGMAX:[0-9]+]] = "ttir.argmax"(%[[FIRST_RESHAPE]]) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<28x3584xf32>) -> tensor<28x1xi32>
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[ARGMAX]]) <{shape = [1 : i32, 28 : i32, 1 : i32]}> : (tensor<28x1xi32>) -> tensor<1x28x1xi32>
    // CHECK: return %[[RESHAPE]] : tensor<1x28x1xi32>
    %1 = "ttir.argmax"(%arg0) <{dim_arg = [0: i32, 2 : i32], keep_dim = true}> : (tensor<128x28x28xf32>) -> tensor<1x28x1xi32>
    return %1 : tensor<1x28x1xi32>
  }

  func.func @argmax_negative_dims(%arg0: tensor<4x5x6xf32>) -> tensor<5xi32> {
    // CHECK-LABEL: func.func @argmax_negative_dims
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0, 2>}> : (tensor<4x5x6xf32>) -> tensor<5x4x6xf32>
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[PERMUTE]]) <{shape = [5 : i32, 24 : i32]}> : (tensor<5x4x6xf32>) -> tensor<5x24xf32>
    // CHECK: %[[ARGMAX:[0-9]+]] = "ttir.argmax"(%[[RESHAPE]]) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<5x24xf32>) -> tensor<5xi32>
    // CHECK: return %[[ARGMAX]] : tensor<5xi32>
    %1 = "ttir.argmax"(%arg0) <{dim_arg = [-3: i32, -1 : i32], keep_dim = false}> : (tensor<4x5x6xf32>) -> tensor<5xi32>
    return %1 : tensor<5xi32>
  }

  func.func public @argmax_5d(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<4x6xi32> {
    // CHECK-LABEL: func.func public @argmax_5d
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0) <{permutation = array<i64: 2, 4, 0, 1, 3>}> : (tensor<2x3x4x5x6xf32>) -> tensor<4x6x2x3x5xf32>
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[PERMUTE]]) <{shape = [4 : i32, 6 : i32, 30 : i32]}> : (tensor<4x6x2x3x5xf32>) -> tensor<4x6x30xf32>
    // CHECK: %[[ARGMAX:[0-9]+]] = "ttir.argmax"(%[[RESHAPE]]) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<4x6x30xf32>) -> tensor<4x6xi32>
    // CHECK: return %[[ARGMAX]] : tensor<4x6xi32>
    %1 = "ttir.argmax"(%arg0) <{dim_arg = [0: i32, 1 : i32, 3 : i32], keep_dim = false}> : (tensor<2x3x4x5x6xf32>) -> tensor<4x6xi32>
    return %1 : tensor<4x6xi32>
  }

  func.func public @argmax_5d_keep(%arg0: tensor<2x3x4x5x6xf32>) -> tensor<1x1x4x1x6xi32> {
    // CHECK-LABEL: func.func public @argmax_5d_keep
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0) <{permutation = array<i64: 2, 4, 0, 1, 3>}> : (tensor<2x3x4x5x6xf32>) -> tensor<4x6x2x3x5xf32>
    // CHECK: %[[RESHAPE_1:[0-9]+]] = "ttir.reshape"(%[[PERMUTE]]) <{shape = [4 : i32, 6 : i32, 30 : i32]}> : (tensor<4x6x2x3x5xf32>) -> tensor<4x6x30xf32>
    // CHECK: %[[ARGMAX:[0-9]+]] = "ttir.argmax"(%[[RESHAPE_1]]) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<4x6x30xf32>) -> tensor<4x6x1xi32>
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[ARGMAX]]) <{shape = [1 : i32, 1 : i32, 4 : i32, 1 : i32, 6 : i32]}> : (tensor<4x6x1xi32>) -> tensor<1x1x4x1x6xi32>
    // CHECK: return %[[RESHAPE]] : tensor<1x1x4x1x6xi32>
    %1 = "ttir.argmax"(%arg0) <{dim_arg = [0: i32, 1 : i32, 3 : i32], keep_dim = true}> : (tensor<2x3x4x5x6xf32>) -> tensor<1x1x4x1x6xi32>
    return %1 : tensor<1x1x4x1x6xi32>
  }
}
