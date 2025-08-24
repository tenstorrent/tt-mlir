// RUN: ttmlir-opt --ttir-explicate-tms -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @binary_same_shape_no_transform(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: "ttir.add"(%arg0, %arg1, %0)
    // CHECK-NOT: "ttir.reshape"
    // CHECK-NOT: "ttir.broadcast"
    return %1 : tensor<64x128xf32>
  }

  func.func @binary_reshape_lhs(%arg0: tensor<64x128xf32>, %arg1: tensor<1x64x128xf32>) -> tensor<1x64x128xf32> {
    %0 = ttir.empty() : tensor<1x64x128xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<1x64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: "ttir.add"([[RESHAPE0]], %arg1, {{%[0-9]+}})
    return %1 : tensor<1x64x128xf32>
  }

  func.func @binary_reshape_rhs(%arg0: tensor<1x64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<1x64x128xf32> {
    %0 = ttir.empty() : tensor<1x64x128xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x64x128xf32>, tensor<64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: "ttir.add"(%arg0, [[RESHAPE0]], {{%[0-9]+}})
    return %1 : tensor<1x64x128xf32>
  }

  func.func @binary_broadcast_lhs(%arg0: tensor<1x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: "ttir.add"([[BCAST0]], %arg1, {{%[0-9]+}})
    return %1 : tensor<64x128xf32>
  }

  func.func @binary_broadcast_rhs(%arg0: tensor<64x128xf32>, %arg1: tensor<1x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<1x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: "ttir.add"(%arg0, [[BCAST0]], {{%[0-9]+}})
    return %1 : tensor<64x128xf32>
  }

  func.func @binary_broadcast_lhs_and_rhs(%arg0: tensor<1x128xf32>, %arg1: tensor<64x1xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x128xf32>, tensor<64x1xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: [[BCAST1:%[0-9]+]] = "ttir.broadcast"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: "ttir.add"([[BCAST0]], [[BCAST1]], {{%[0-9]+}})
    return %1 : tensor<64x128xf32>
  }

  func.func @binary_broadcast_and_reshape_lhs_scalar(%arg0: tensor<f32>, %arg1: tensor<32x64x128xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<f32>, tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x1xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"([[RESHAPE0]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: "ttir.add"([[BCAST0]], %arg1, {{%[0-9]+}})
    return %1 : tensor<32x64x128xf32>
  }

  func.func @binary_broadcast_and_reshape_rhs_scalar(%arg0: tensor<32x64x128xf32>, %arg1: tensor<f32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x64x128xf32>, tensor<f32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x1xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"([[RESHAPE0]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: "ttir.add"(%arg0, [[BCAST0]], {{%[0-9]+}})
    return %1 : tensor<32x64x128xf32>
  }

  func.func @binary_broadcast_and_reshape_lhs_and_broadcast_rhs(%arg0: tensor<1x128xf32>, %arg1: tensor<32x64x1xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x128xf32>, tensor<32x64x1xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"([[RESHAPE0]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: [[BCAST1:%[0-9]+]] = "ttir.broadcast"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: "ttir.add"([[BCAST0]], [[BCAST1]], {{%[0-9]+}})
    return %1 : tensor<32x64x128xf32>
  }

  func.func @binary_broadcast_and_reshape_rhs_and_broadcast_lhs(%arg0: tensor<32x64x1xf32>, %arg1: tensor<1x128xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<32x64x1xf32>, tensor<1x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: [[BCAST1:%[0-9]+]] = "ttir.broadcast"([[RESHAPE0]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: "ttir.add"([[BCAST0]], [[BCAST1]], {{%[0-9]+}})
    return %1 : tensor<32x64x128xf32>
  }

  func.func @binary_chained_operations(%arg0: tensor<f32>, %arg1: tensor<1x128xf32>, %arg2: tensor<64x1xf32>, %arg3: tensor<f32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<1x128xf32>
    %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<f32>, tensor<1x128xf32>, tensor<1x128xf32>) -> tensor<1x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"([[RESHAPE0]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x128xf32>
    // CHECK: [[ADD0:%[0-9]+]] = "ttir.add"([[BCAST0]], %arg1, {{%[0-9]+}})

    %2 = ttir.empty() : tensor<64x1xf32>
    %3 = "ttir.add"(%arg2, %arg3, %2) : (tensor<64x1xf32>, tensor<f32>, tensor<64x1xf32>) -> tensor<64x1xf32>
    // CHECK: [[RESHAPE1:%[0-9]+]] = "ttir.reshape"(%arg3, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1xf32>
    // CHECK: [[BCAST1:%[0-9]+]] = "ttir.broadcast"([[RESHAPE1]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x1xf32>
    // CHECK: [[ADD1:%[0-9]+]] = "ttir.add"(%arg2, [[BCAST1]], {{%[0-9]+}})

    %4 = ttir.empty() : tensor<64x128xf32>
    %5 = "ttir.multiply"(%1, %3, %4) : (tensor<1x128xf32>, tensor<64x1xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: [[BCAST2:%[0-9]+]] = "ttir.broadcast"([[ADD0]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: [[BCAST3:%[0-9]+]] = "ttir.broadcast"([[ADD1]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: "ttir.multiply"([[BCAST2]], [[BCAST3]], {{%[0-9]+}})

    return %5 : tensor<64x128xf32>
  }
}
