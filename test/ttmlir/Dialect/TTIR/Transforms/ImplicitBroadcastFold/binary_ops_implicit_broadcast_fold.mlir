// RUN: ttmlir-opt --ttir-implicit-broadcast-fold -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
  func.func @binary_implicit_broadcast(%arg0: tensor<1x128xf32>, %arg1: tensor<64x1xf32>) -> tensor<64x128xf32> {
    %1 = "ttir.add"(%arg0, %arg1) : (tensor<1x128xf32>, tensor<64x1xf32>) -> tensor<64x128xf32>
    // CHECK-NOT: "ttir.broadcast"
    // CHECK: "ttir.add"(%arg0, %arg1)
    // CHECK-SAME: -> tensor<64x128xf32>
    return %1 : tensor<64x128xf32>
  }

  func.func @binary_explicit_broadcast_lhs(%arg0: tensor<1x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 64, 1>}> : (tensor<1x128xf32>) -> tensor<64x128xf32>
    %3 = "ttir.add"(%1, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK-NOT: "ttir.broadcast"
    // CHECK: "ttir.add"(%arg0, %arg1)
    // CHECK-SAME: -> tensor<64x128xf32>
    return %3 : tensor<64x128xf32>
  }

  func.func @binary_explicit_broadcast_rhs(%arg0: tensor<64x128xf32>, %arg1: tensor<1x128xf32>) -> tensor<64x128xf32> {
    %1 = "ttir.broadcast"(%arg1) <{broadcast_dimensions = array<i64: 64, 1>}> : (tensor<1x128xf32>) -> tensor<64x128xf32>
    %3 = "ttir.add"(%arg0, %1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK-NOT: "ttir.broadcast"
    // CHECK: "ttir.add"(%arg0, %arg1)
    // CHECK-SAME: -> tensor<64x128xf32>
    return %3 : tensor<64x128xf32>
  }

  func.func @binary_explicit_broadcast_lhs_and_rhs_first(%arg0: tensor<1x128xf32>, %arg1: tensor<1x128xf32>) -> tensor<64x128xf32> {
    %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 64, 1>}> : (tensor<1x128xf32>) -> tensor<64x128xf32>
    %3 = "ttir.broadcast"(%arg1) <{broadcast_dimensions = array<i64: 64, 1>}> : (tensor<1x128xf32>) -> tensor<64x128xf32>
    %5 = "ttir.add"(%1, %3) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: [[ADD:%[0-9]+]] = "ttir.add"(%arg0, %arg1)
    // CHECK-SAME: -> tensor<1x128xf32>
    // CHECK: "ttir.broadcast"([[ADD]]
    // CHECK-SAME: -> tensor<64x128xf32>
    return %5 : tensor<64x128xf32>
  }

  func.func @binary_explicit_broadcast_lhs_and_rhs_second(%arg0: tensor<32x1x1xf32>, %arg1: tensor<1x1x1xf32>) -> tensor<32x64x128xf32> {
    %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 64, 128>}> : (tensor<32x1x1xf32>) -> tensor<32x64x128xf32>
    %3 = "ttir.broadcast"(%arg1) <{broadcast_dimensions = array<i64: 32, 64, 128>}> : (tensor<1x1x1xf32>) -> tensor<32x64x128xf32>
    %5 = "ttir.add"(%1, %3) : (tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[ADD:%[0-9]+]] = "ttir.add"(%arg0, %arg1)
    // CHECK-SAME: -> tensor<32x1x1xf32>
    // CHECK: "ttir.broadcast"([[ADD]]
    // CHECK-SAME: -> tensor<32x64x128xf32>
    return %5 : tensor<32x64x128xf32>
  }

  func.func @binary_chained_operations(%arg0: tensor<1x1x128xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x128xf32>, %arg3: tensor<1x1x128xf32>) -> tensor<32x64x128xf32> {
    %1 = "ttir.broadcast"(%arg0) <{broadcast_dimensions = array<i64: 1, 64, 1>}> : (tensor<1x1x128xf32>) -> tensor<1x64x128xf32>
    %3 = "ttir.broadcast"(%arg1) <{broadcast_dimensions = array<i64: 1, 64, 128>}> : (tensor<1x1x1xf32>) -> tensor<1x64x128xf32>
    %5 = "ttir.add"(%1, %3) : (tensor<1x64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>

    %7 = "ttir.broadcast"(%arg2) <{broadcast_dimensions = array<i64: 32, 64, 1>}> : (tensor<1x1x128xf32>) -> tensor<32x64x128xf32>
    %9 = "ttir.broadcast"(%arg3) <{broadcast_dimensions = array<i64: 32, 64, 1>}> : (tensor<1x1x128xf32>) -> tensor<32x64x128xf32>
    %11 = "ttir.multiply"(%7, %9) : (tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>

    %13 = "ttir.broadcast"(%5) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x64x128xf32>) -> tensor<32x64x128xf32>
    %15 = "ttir.add"(%13, %11) : (tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[ADD0:%[0-9]+]] = "ttir.add"(%arg0, %arg1)
    // CHECK-SAME: -> tensor<1x1x128xf32>
    // CHECK: [[MUL0:%[0-9]+]] = "ttir.multiply"(%arg2, %arg3
    // CHECK-SAME: -> tensor<1x1x128xf32>
    // CHECK: [[ADD1:%[0-9]+]] = "ttir.add"([[ADD0]], [[MUL0]]
    // CHECK-SAME: -> tensor<1x1x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"([[ADD1]]
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: "ttir.broadcast"([[BCAST0]]
    // CHECK-SAME: -> tensor<32x64x128xf32>
    return %15 : tensor<32x64x128xf32>
  }
}
