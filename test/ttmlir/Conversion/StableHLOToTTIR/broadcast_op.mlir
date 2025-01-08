// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline %s | FileCheck %s
module {
  func.func public @main(%arg0: tensor<1xf32> {mhlo.layout_mode = "default"}, %arg1: tensor<512x512xf32> {mhlo.layout_mode = "default"}) -> (tensor<512x512xf32> {jax.result_info = "", mhlo.layout_mode = "default"}) {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<1xf32>) -> tensor<512x512xf32>
    %1 = stablehlo.maximum %0, %arg1 : tensor<512x512xf32>
    // CHECK: %{{.*}} = "ttir.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32]
    // CHECK: %{{.*}} = "ttir.broadcast"
    // CHECK-SAME: broadcast_dimensions = array<i32: 512, 512>
    return %1 : tensor<512x512xf32>
  }
}

module {
  func.func @main(%arg0: tensor<1x23x40x1xf32>, %arg1: tensor<128xf32>) -> tensor<1x23x40x128xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1, 2, 3] : (tensor<1x23x40x1xf32>) -> tensor<1x23x40x128xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [3] : (tensor<128xf32>) -> tensor<1x23x40x128xf32>
    // CHECK: %{{.*}} = "ttir.broadcast"
    // CHECK-SAME: broadcast_dimensions = array<i32: 1, 1, 1, 128>
    // CHECK: %{{.*}} = "ttir.reshape"
    // CHECK-SAME: shape = [1 : i32, 1 : i32, 1 : i32, 128 : i32]
    // CHECK: %{{.*}} = "ttir.broadcast"
    // CHECK-SAME: broadcast_dimensions = array<i32: 1, 23, 40, 1>
    %2 = stablehlo.divide %0, %1 : tensor<1x23x40x128xf32>
    return %2 : tensor<1x23x40x128xf32>
  }
}

module {
  func.func @main(%arg0: tensor<32xi64>, %arg1: tensor<32x1xi64>) -> tensor<32x32xi1> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [1] : (tensor<32xi64>) -> tensor<32x32xi64>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<32x1xi64>) -> tensor<32x32xi64>
    %2 = stablehlo.compare  GT, %0, %1,  SIGNED : (tensor<32x32xi64>, tensor<32x32xi64>) -> tensor<32x32xi1>
    // CHECK: %{{.*}} = "ttir.reshape"
    // CHECK-SAME: shape = [1 : i32, 32 : i32]
    // CHECK: %{{.*}} = "ttir.broadcast"
    // CHECK-SAME: broadcast_dimensions = array<i32: 32, 1>
    // CHECK: %{{.*}} = "ttir.broadcast"
    // CHECK-SAME: broadcast_dimensions = array<i32: 1, 32>
    return %2 : tensor<32x32xi1>
  }
}

module {
  func.func @main(%arg0: tensor<16x1xf32>, %arg1: tensor<1x1x32xi64>) -> tensor<1x16x32xf32> {
    %0 = stablehlo.convert %arg1 : (tensor<1x1x32xi64>) -> tensor<1x1x32xf32>
    %1 = stablehlo.broadcast_in_dim %arg0, dims = [1, 2] : (tensor<16x1xf32>) -> tensor<1x16x32xf32>
    %2 = stablehlo.broadcast_in_dim %0, dims = [0, 1, 2] : (tensor<1x1x32xf32>) -> tensor<1x16x32xf32>
    %3 = stablehlo.multiply %1, %2 : tensor<1x16x32xf32>
    // CHECK: %{{.*}} = "ttir.reshape"
    // CHECK-SAME: shape = [1 : i32, 16 : i32, 1 : i32]
    // CHECK: %{{.*}} = "ttir.broadcast"
    // CHECK-SAME: broadcast_dimensions = array<i32: 1, 1, 32>
    // CHECK: %{{.*}} = "ttir.broadcast"
    // CHECK-SAME: broadcast_dimensions = array<i32: 1, 16, 1>
    return %3 : tensor<1x16x32xf32>
  }
}

module {
  func.func @main(%arg0: tensor<1x10xi64>, %arg1: tensor<10x1xi64>) -> tensor<10x10xi64> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0, 1] : (tensor<1x10xi64>) -> tensor<10x10xi64>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [0, 1] : (tensor<10x1xi64>) -> tensor<10x10xi64>
    %2 = stablehlo.subtract %0, %1 : tensor<10x10xi64>
    // CHECK: %{{.*}} = "ttir.broadcast"
    // CHECK-SAME: broadcast_dimensions = array<i32: 10, 1>
    // CHECK: %{{.*}} = "ttir.broadcast"
    // CHECK-SAME: broadcast_dimensions = array<i32: 1, 10>
    return %2 : tensor<10x10xi64>
  }
}

module {
  func.func @main(%arg0: tensor<8xf32>, %arg1: tensor<1xf32>) -> tensor<8xf32> {
    %0 = stablehlo.broadcast_in_dim %arg0, dims = [0] : (tensor<8xf32>) -> tensor<8xf32>
    %1 = stablehlo.broadcast_in_dim %arg1, dims = [0] : (tensor<1xf32>) -> tensor<8xf32>
    %2 = stablehlo.add %0, %1 : tensor<8xf32>
    // CHECK: %{{.*}} = "ttir.broadcast"
    // CHECK-SAME: broadcast_dimensions = array<i32: 1>
    // CHECK: %{{.*}} = "ttir.broadcast"
    // CHECK-SAME: broadcast_dimensions = array<i32: 8>
    return %2 : tensor<8xf32>
  }
}

module {
  func.func @main(%arg0: tensor<6x2xf32>) -> tensor<2400x2xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<6x2xf32>) -> tensor<1x6x2xf32>
    %1 = stablehlo.reshape %0 : (tensor<1x6x2xf32>) -> tensor<1x6x1x2xf32>
    %2 = stablehlo.broadcast_in_dim %1, dims = [0, 1, 2, 3] : (tensor<1x6x1x2xf32>) -> tensor<400x6x1x2xf32>
    %3 = stablehlo.reshape %2 : (tensor<400x6x1x2xf32>) -> tensor<2400x1x2xf32>
    %4 = stablehlo.reshape %3 : (tensor<2400x1x2xf32>) -> tensor<2400x2xf32>
    // CHECK: %{{.*}} = "ttir.broadcast"
    // CHECK-SAME: broadcast_dimensions = array<i32: 400, 1, 1, 1>
    return %4 : tensor<2400x2xf32>
  }
}
