// RUN: ttmlir-opt --ttir-explicate-tms -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @ternary_same_shape_no_transform(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: "ttir.where"(%arg0, %arg1, %arg2, %0)
    // CHECK-NOT: "ttir.reshape"
    // CHECK-NOT: "ttir.broadcast"
    return %1 : tensor<64x128xf32>
  }

  func.func @ternary_reshape_first(%arg0: tensor<64x128xf32>, %arg1: tensor<1x64x128xf32>, %arg2: tensor<1x64x128xf32>) -> tensor<1x64x128xf32> {
    %0 = ttir.empty() : tensor<1x64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<64x128xf32>, tensor<1x64x128xf32>, tensor<1x64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: "ttir.where"([[RESHAPE0]], %arg1, %arg2, {{%[0-9]+}})
    return %1 : tensor<1x64x128xf32>
  }

  func.func @ternary_reshape_second(%arg0: tensor<1x64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<1x64x128xf32>) -> tensor<1x64x128xf32> {
    %0 = ttir.empty() : tensor<1x64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<1x64x128xf32>, tensor<64x128xf32>, tensor<1x64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: "ttir.where"(%arg0, [[RESHAPE0]], %arg2, {{%[0-9]+}})
    return %1 : tensor<1x64x128xf32>
  }

  func.func @ternary_reshape_third(%arg0: tensor<1x64x128xf32>, %arg1: tensor<1x64x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<1x64x128xf32> {
    %0 = ttir.empty() : tensor<1x64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<1x64x128xf32>, tensor<1x64x128xf32>, tensor<64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg2, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: "ttir.where"(%arg0, %arg1, [[RESHAPE0]], {{%[0-9]+}})
    return %1 : tensor<1x64x128xf32>
  }

  func.func @ternary_reshape_first_second(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<1x64x128xf32>) -> tensor<1x64x128xf32> {
    %0 = ttir.empty() : tensor<1x64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<1x64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: [[RESHAPE1:%[0-9]+]] = "ttir.reshape"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: "ttir.where"([[RESHAPE0]], [[RESHAPE1]], %arg2, {{%[0-9]+}})
    return %1 : tensor<1x64x128xf32>
  }

  func.func @ternary_reshape_second_third(%arg0: tensor<1x64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<1x64x128xf32> {
    %0 = ttir.empty() : tensor<1x64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<1x64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: [[RESHAPE1:%[0-9]+]] = "ttir.reshape"(%arg2, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: "ttir.where"(%arg0, [[RESHAPE0]], [[RESHAPE1]], {{%[0-9]+}})
    return %1 : tensor<1x64x128xf32>
  }

  func.func @ternary_reshape_first_third(%arg0: tensor<64x128xf32>, %arg1: tensor<1x64x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<1x64x128xf32> {
    %0 = ttir.empty() : tensor<1x64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<64x128xf32>, tensor<1x64x128xf32>, tensor<64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: [[RESHAPE1:%[0-9]+]] = "ttir.reshape"(%arg2, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: "ttir.where"([[RESHAPE0]], %arg1, [[RESHAPE1]], {{%[0-9]+}})
    return %1 : tensor<1x64x128xf32>
  }

  func.func @ternary_broadcast_first(%arg0: tensor<1x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<1x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: "ttir.where"([[BCAST0]], %arg1, %arg2, {{%[0-9]+}})
    return %1 : tensor<64x128xf32>
  }

  func.func @ternary_broadcast_second(%arg0: tensor<64x128xf32>, %arg1: tensor<1x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<64x128xf32>, tensor<1x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: "ttir.where"(%arg0, [[BCAST0]], %arg2, {{%[0-9]+}})
    return %1 : tensor<64x128xf32>
  }

  func.func @ternary_broadcast_third(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<1x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<1x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"(%arg2, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: "ttir.where"(%arg0, %arg1, [[BCAST0]], {{%[0-9]+}})
    return %1 : tensor<64x128xf32>
  }

  func.func @ternary_broadcast_first_second(%arg0: tensor<1x128xf32>, %arg1: tensor<1x128xf32>, %arg2: tensor<64x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<1x128xf32>, tensor<1x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: [[BCAST1:%[0-9]+]] = "ttir.broadcast"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: "ttir.where"([[BCAST0]], [[BCAST1]], %arg2, {{%[0-9]+}})
    return %1 : tensor<64x128xf32>
  }

  func.func @ternary_broadcast_first_third(%arg0: tensor<1x128xf32>, %arg1: tensor<64x128xf32>, %arg2: tensor<1x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<1x128xf32>, tensor<64x128xf32>, tensor<1x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: [[BCAST1:%[0-9]+]] = "ttir.broadcast"(%arg2, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: "ttir.where"([[BCAST0]], %arg1, [[BCAST1]], {{%[0-9]+}})
    return %1 : tensor<64x128xf32>
  }

  func.func @ternary_broadcast_second_third(%arg0: tensor<64x128xf32>, %arg1: tensor<1x128xf32>, %arg2: tensor<1x128xf32>) -> tensor<64x128xf32> {
    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<64x128xf32>, tensor<1x128xf32>, tensor<1x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: [[BCAST1:%[0-9]+]] = "ttir.broadcast"(%arg2, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<64x128xf32>
    // CHECK: "ttir.where"(%arg0, [[BCAST0]], [[BCAST1]], {{%[0-9]+}})
    return %1 : tensor<64x128xf32>
  }

  func.func @ternary_reshape_and_broadcast_first_second(%arg0: tensor<f32>, %arg1: tensor<1x128xf32>, %arg2: tensor<32x64x128xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<f32>, tensor<1x128xf32>, tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x1xf32>
    // CHECK: [[RESHAPE1:%[0-9]+]] = "ttir.reshape"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"([[RESHAPE0]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: [[BCAST1:%[0-9]+]] = "ttir.broadcast"([[RESHAPE1]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: "ttir.where"([[BCAST0]], [[BCAST1]], %arg2, {{%[0-9]+}})
    return %1 : tensor<32x64x128xf32>
  }

  func.func @ternary_reshape_and_broadcast_first_third(%arg0: tensor<1x128xf32>, %arg1: tensor<32x64x128xf32>, %arg2: tensor<f32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<1x128xf32>, tensor<32x64x128xf32>, tensor<f32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg0, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x128xf32>
    // CHECK: [[RESHAPE1:%[0-9]+]] = "ttir.reshape"(%arg2, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x1xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"([[RESHAPE0]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: [[BCAST1:%[0-9]+]] = "ttir.broadcast"([[RESHAPE1]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: "ttir.where"([[BCAST0]], %arg1, [[BCAST1]], {{%[0-9]+}})
    return %1 : tensor<32x64x128xf32>
  }

  func.func @ternary_reshape_and_broadcast_second_third(%arg0: tensor<32x64x128xf32>, %arg1: tensor<f32>, %arg2: tensor<1x128xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<32x64x128xf32>, tensor<f32>, tensor<1x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[RESHAPE0:%[0-9]+]] = "ttir.reshape"(%arg1, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x1xf32>
    // CHECK: [[RESHAPE1:%[0-9]+]] = "ttir.reshape"(%arg2, {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"([[RESHAPE0]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: [[BCAST1:%[0-9]+]] = "ttir.broadcast"([[RESHAPE1]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: "ttir.where"(%arg0, [[BCAST0]], [[BCAST1]], {{%[0-9]+}})
    return %1 : tensor<32x64x128xf32>
  }

  func.func @ternary_chained_operations(%arg0: tensor<f32>, %arg1: tensor<1x128xf32>, %arg2: tensor<64x1xf32>, %arg3: tensor<f32>, %arg4: tensor<32x64x128xf32>) -> tensor<32x64x128xf32> {
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

    %4 = ttir.empty() : tensor<32x64x128xf32>
    %5 = "ttir.where"(%1, %3, %arg4, %4) : (tensor<1x128xf32>, tensor<64x1xf32>, tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[RESHAPE2:%[0-9]+]] = "ttir.reshape"([[ADD0]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x128xf32>
    // CHECK: [[RESHAPE3:%[0-9]+]] = "ttir.reshape"([[ADD1]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x1xf32>
    // CHECK: [[BCAST2:%[0-9]+]] = "ttir.broadcast"([[RESHAPE2]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: [[BCAST3:%[0-9]+]] = "ttir.broadcast"([[RESHAPE3]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: [[WHERE0:%[0-9]+]] = "ttir.where"([[BCAST2]], [[BCAST3]], %arg4, {{%[0-9]+}})

    %6 = ttir.empty() : tensor<32x64x128xf32>
    %7 = "ttir.where"(%1, %3, %5, %6) : (tensor<1x128xf32>, tensor<64x1xf32>, tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[RESHAPE3:%[0-9]+]] = "ttir.reshape"([[ADD0]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x1x128xf32>
    // CHECK: [[RESHAPE4:%[0-9]+]] = "ttir.reshape"([[ADD1]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x1xf32>
    // CHECK: [[BCAST3:%[0-9]+]] = "ttir.broadcast"([[RESHAPE3]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: [[BCAST4:%[0-9]+]] = "ttir.broadcast"([[RESHAPE4]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    // CHECK: "ttir.where"([[BCAST3]], [[BCAST4]], [[WHERE0]], {{%[0-9]+}})

    return %7 : tensor<32x64x128xf32>
  }
}
