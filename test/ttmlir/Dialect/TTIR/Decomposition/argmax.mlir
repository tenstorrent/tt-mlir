// RUN: ttmlir-opt --ttir-to-ttir-decomposition -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  func.func public @argmax_3d(%arg0: tensor<128x28x28xf32>) -> tensor<28xi32> {
    // CHECK-LABEL: func.func public @argmax_3d
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0, 2>}> : (tensor<128x28x28xf32>) -> tensor<28x128x28xf32>
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0) <{shape = [28 : i32, 3584 : i32]}> : (tensor<128x28x28xf32>) -> tensor<28x3584xf32>
    // CHECK: %[[ARGMAX:[0-9]+]] = "ttir.argmax"(%[[RESHAPE]]) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<28x3584xf32>) -> tensor<28xi32>
    // CHECK: return %[[ARGMAX]] : tensor<28xi32>
    %1 = "ttir.argmax"(%arg0) <{dim_arg = [0: i32, 2 : i32], keep_dim = false}> : (tensor<128x28x28xf32>) -> tensor<28xi32>
    return %1 : tensor<28xi32>
  }

  func.func public @argmax_3d_keep(%arg0: tensor<128x28x28xf32>) -> tensor<1x28x1xi32> {
    // CHECK-LABEL: func.func public @argmax_3d_keep
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0, 2>}> : (tensor<128x28x28xf32>) -> tensor<28x128x28xf32>
    // CHECK: %[[FIRST_RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0) <{shape = [28 : i32, 3584 : i32]}> : (tensor<128x28x28xf32>) -> tensor<28x3584xf32>
    // CHECK: %[[ARGMAX:[0-9]+]] = "ttir.argmax"(%[[FIRST_RESHAPE]]) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<28x3584xf32>) -> tensor<28x1xi32>
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%[[ARGMAX]]) <{shape = [1 : i32, 28 : i32, 1 : i32]}> : (tensor<28x1xi32>) -> tensor<1x28x1xi32>
    // CHECK: return %[[RESHAPE]] : tensor<1x28x1xi32>
    %1 = "ttir.argmax"(%arg0) <{dim_arg = [0: i32, 2 : i32], keep_dim = true}> : (tensor<128x28x28xf32>) -> tensor<1x28x1xi32>
    return %1 : tensor<1x28x1xi32>
  }

  func.func @argmax_negative_dims(%arg0: tensor<4x5x6xf32>) -> tensor<5xi32> {
    // CHECK-LABEL: func.func @argmax_negative_dims
    // CHECK: %[[PERMUTE:[0-9]+]] = "ttir.permute"(%arg0) <{permutation = array<i64: 1, 0, 2>}> : (tensor<4x5x6xf32>) -> tensor<5x4x6xf32>
    // CHECK: %[[RESHAPE:[0-9]+]] = "ttir.reshape"(%arg0) <{shape = [5 : i32, 24 : i32]}> : (tensor<4x5x6xf32>) -> tensor<5x24xf32>
    // CHECK: %[[ARGMAX:[0-9]+]] = "ttir.argmax"(%[[RESHAPE]]) <{dim_arg = [1 : i32], keep_dim = false}> : (tensor<5x24xf32>) -> tensor<5xi32>
    // CHECK: return %[[ARGMAX]] : tensor<5xi32>
    %1 = "ttir.argmax"(%arg0) <{dim_arg = [-3: i32, -1 : i32], keep_dim = false}> : (tensor<4x5x6xf32>) -> tensor<5xi32>
    return %1 : tensor<5xi32>
  }
}
