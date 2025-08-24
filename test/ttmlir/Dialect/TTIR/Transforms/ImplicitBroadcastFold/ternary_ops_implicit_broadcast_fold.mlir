// RUN: ttmlir-opt --ttir-implicit-broadcast-fold -o %t %s
// RUN: FileCheck --input-file=%t %s

module {
  func.func @ternary_implicit_broadcast(%arg0: tensor<1x1x128xf32>, %arg1: tensor<1x64x1xf32>, %arg2: tensor<32x1x1xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.where"(%arg0, %arg1, %arg2, %0) : (tensor<1x1x128xf32>, tensor<1x64x1xf32>, tensor<32x1x1xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK-NOT: "ttir.broadcast"
    // CHECK: "ttir.where"(%arg0, %arg1, %arg2, %0)
    // CHECK-SAME: -> tensor<32x64x128xf32>
    return %1 : tensor<32x64x128xf32>
  }

  func.func @ternary_explicit_broadcast_first_operand(%arg0: tensor<1x1x128xf32>, %arg1: tensor<1x64x1xf32>, %arg2: tensor<32x1x1xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 32, 64, 1>}> : (tensor<1x1x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %2 = ttir.empty() : tensor<32x64x128xf32>
    %3 = "ttir.where"(%1, %arg1, %arg2, %2) : (tensor<32x64x128xf32>, tensor<1x64x1xf32>, tensor<32x1x1xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK-NOT: "ttir.broadcast"
    // CHECK: "ttir.where"(%arg0, %arg1, %arg2, %0)
    // CHECK-SAME: -> tensor<32x64x128xf32>
    return %3 : tensor<32x64x128xf32>
  }

  func.func @ternary_explicit_broadcast_second_operand(%arg0: tensor<1x1x128xf32>, %arg1: tensor<1x64x1xf32>, %arg2: tensor<32x1x1xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i64: 32, 1, 128>}> : (tensor<1x64x1xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %2 = ttir.empty() : tensor<32x64x128xf32>
    %3 = "ttir.where"(%arg0, %1, %arg2, %2) : (tensor<1x1x128xf32>, tensor<32x64x128xf32>, tensor<32x1x1xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK-NOT: "ttir.broadcast"
    // CHECK: "ttir.where"(%arg0, %arg1, %arg2, %0)
    // CHECK-SAME: -> tensor<32x64x128xf32>
    return %3 : tensor<32x64x128xf32>
  }

  func.func @ternary_explicit_broadcast_third_operand(%arg0: tensor<1x1x128xf32>, %arg1: tensor<1x64x1xf32>, %arg2: tensor<32x1x1xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.broadcast"(%arg2, %0) <{broadcast_dimensions = array<i64: 1, 64, 128>}> : (tensor<32x1x1xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %2 = ttir.empty() : tensor<32x64x128xf32>
    %3 = "ttir.where"(%arg0, %arg1, %1, %2) : (tensor<1x1x128xf32>, tensor<1x64x1xf32>, tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK-NOT: "ttir.broadcast"
    // CHECK: "ttir.where"(%arg0, %arg1, %arg2, %0)
    // CHECK-SAME: -> tensor<32x64x128xf32>
    return %3 : tensor<32x64x128xf32>
  }

  func.func @ternary_explicit_broadcast_first_and_second_operand(%arg0: tensor<1x1x128xf32>, %arg1: tensor<1x1x128xf32>, %arg2: tensor<32x1x1xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 32, 64, 1>}> : (tensor<1x1x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %2 = ttir.empty() : tensor<32x64x128xf32>
    %3 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i64: 32, 64, 1>}> : (tensor<1x1x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %4 = ttir.empty() : tensor<32x64x128xf32>
    %5 = "ttir.where"(%1, %3, %arg2, %4) : (tensor<32x64x128xf32>, tensor<32x64x128xf32>, tensor<32x1x1xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[WHERE:%[0-9]+]] = "ttir.where"(%arg0, %arg1, %arg2, %0)
    // CHECK-SAME: -> tensor<32x1x128xf32>
    // CHECK: "ttir.broadcast"([[WHERE]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    return %5 : tensor<32x64x128xf32>
  }

  func.func @ternary_explicit_broadcast_first_and_third_operand(%arg0: tensor<1x1x128xf32>, %arg1: tensor<1x64x1xf32>, %arg2: tensor<1x1x128xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 32, 64, 1>}> : (tensor<1x1x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %2 = ttir.empty() : tensor<32x64x128xf32>
    %3 = "ttir.broadcast"(%arg2, %0) <{broadcast_dimensions = array<i64: 32, 64, 1>}> : (tensor<1x1x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %4 = ttir.empty() : tensor<32x64x128xf32>
    %5 = "ttir.where"(%1, %arg1, %3, %4) : (tensor<32x64x128xf32>, tensor<1x64x1xf32>, tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[WHERE:%[0-9]+]] = "ttir.where"(%arg0, %arg1, %arg2, %0)
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: "ttir.broadcast"([[WHERE]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    return %5 : tensor<32x64x128xf32>
  }

  func.func @ternary_explicit_broadcast_second_and_third_operand(%arg0: tensor<1x1x128xf32>, %arg1: tensor<1x64x1xf32>, %arg2: tensor<32x1x1xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.broadcast"(%arg1, %0) <{broadcast_dimensions = array<i64: 32, 1, 128>}> : (tensor<1x64x1xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %2 = ttir.empty() : tensor<32x64x128xf32>
    %3 = "ttir.broadcast"(%arg2, %0) <{broadcast_dimensions = array<i64: 1, 64, 128>}> : (tensor<32x1x1xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %4 = ttir.empty() : tensor<32x64x128xf32>
    %5 = "ttir.where"(%arg0, %1, %3, %4) : (tensor<1x1x128xf32>, tensor<32x64x128xf32>, tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[WHERE:%[0-9]+]] = "ttir.where"(%arg0, %arg1, %arg2, %0)
    // CHECK-SAME: -> tensor<32x64x128xf32>
    return %5 : tensor<32x64x128xf32>
  }

  func.func @ternary_explicit_broadcast_all_operands(%arg0: tensor<1x1x1xf32>, %arg1: tensor<1x1x128xf32>, %arg2: tensor<1x1x128xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<32x64x128xf32>
    %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 32, 64, 128>}> : (tensor<1x1x1xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %2 = ttir.empty() : tensor<32x64x128xf32>
    %3 = "ttir.broadcast"(%arg1, %2) <{broadcast_dimensions = array<i64: 32, 64, 1>}> : (tensor<1x1x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %4 = ttir.empty() : tensor<32x64x128xf32>
    %5 = "ttir.broadcast"(%arg2, %4) <{broadcast_dimensions = array<i64: 32, 64, 1>}> : (tensor<1x1x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %6 = ttir.empty() : tensor<32x64x128xf32>
    %7 = "ttir.where"(%1, %3, %5, %6) : (tensor<32x64x128xf32>, tensor<32x64x128xf32>, tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[WHERE:%[0-9]+]] = "ttir.where"(%arg0, %arg1, %arg2, %0)
    // CHECK-SAME: -> tensor<1x1x128xf32>
    // CHECK: "ttir.broadcast"([[WHERE]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    return %7 : tensor<32x64x128xf32>
  }

  func.func @ternary_chained_operations(%arg0: tensor<1x1x128xf32>, %arg1: tensor<1x1x1xf32>, %arg2: tensor<1x1x128xf32>, %arg3: tensor<1x1x1xf32>, %arg4: tensor<1x1x1xf32>, %arg5: tensor<1x1x1xf32>, %arg6: tensor<1x1x1xf32>, %arg7: tensor<1x1x1xf32>, %arg8: tensor<1x1x1xf32>) -> tensor<32x64x128xf32> {
    %0 = ttir.empty() : tensor<1x64x128xf32>
    %1 = "ttir.broadcast"(%arg0, %0) <{broadcast_dimensions = array<i64: 1, 64, 1>}> : (tensor<1x1x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    %2 = ttir.empty() : tensor<1x64x128xf32>
    %3 = "ttir.broadcast"(%arg1, %2) <{broadcast_dimensions = array<i64: 1, 64, 128>}> : (tensor<1x1x1xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    %4 = ttir.empty() : tensor<1x64x128xf32>
    %5 = "ttir.broadcast"(%arg2, %4) <{broadcast_dimensions = array<i64: 1, 64, 1>}> : (tensor<1x1x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    %6 = ttir.empty() : tensor<1x64x128xf32>
    %7 = "ttir.where"(%1, %3, %5, %6) : (tensor<1x64x128xf32>, tensor<1x64x128xf32>, tensor<1x64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>

    %8 = ttir.empty() : tensor<1x1x128xf32>
    %9 = "ttir.broadcast"(%arg3, %8) <{broadcast_dimensions = array<i64: 1, 1, 128>}> : (tensor<1x1x1xf32>, tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %10 = ttir.empty() : tensor<1x1x128xf32>
    %11 = "ttir.broadcast"(%arg4, %10) <{broadcast_dimensions = array<i64: 1, 1, 128>}> : (tensor<1x1x1xf32>, tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %12 = ttir.empty() : tensor<1x1x128xf32>
    %13 = "ttir.broadcast"(%arg5, %12) <{broadcast_dimensions = array<i64: 1, 1, 128>}> : (tensor<1x1x1xf32>, tensor<1x1x128xf32>) -> tensor<1x1x128xf32>
    %14 = ttir.empty() : tensor<1x1x128xf32>
    %15 = "ttir.where"(%9, %11, %13, %14) : (tensor<1x1x128xf32>, tensor<1x1x128xf32>, tensor<1x1x128xf32>, tensor<1x1x128xf32>) -> tensor<1x1x128xf32>

    %16 = ttir.empty() : tensor<1x64x128xf32>
    %17 = "ttir.broadcast"(%arg6, %16) <{broadcast_dimensions = array<i64: 1, 64, 128>}> : (tensor<1x1x1xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    %18 = ttir.empty() : tensor<1x64x128xf32>
    %23 = "ttir.where"(%17, %arg7, %arg8, %18) : (tensor<1x64x128xf32>, tensor<1x1x1xf32>, tensor<1x1x1xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>

    %24 = ttir.empty() : tensor<32x64x128xf32>
    %25 = "ttir.broadcast"(%23, %24) <{broadcast_dimensions = array<i64: 32, 1, 1>}> : (tensor<1x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    %26 = ttir.empty() : tensor<32x64x128xf32>
    %27 = "ttir.where"(%7, %15, %25, %24) : (tensor<1x64x128xf32>, tensor<1x1x128xf32>, tensor<32x64x128xf32>, tensor<32x64x128xf32>) -> tensor<32x64x128xf32>
    // CHECK: [[WHERE0:%[0-9]+]] = "ttir.where"(%arg0, %arg1, %arg2, %0)
    // CHECK-SAME: -> tensor<1x1x128xf32>
    // CHECK: [[WHERE1:%[0-9]+]] = "ttir.where"(%arg3, %arg4, %arg5, %2)
    // CHECK-SAME: -> tensor<1x1x1xf32>
    // CHECK: [[WHERE2:%[0-9]+]] = "ttir.where"(%arg6, %arg7, %arg8, %4)
    // CHECK-SAME: -> tensor<1x1x1xf32>
    // CHECK: [[WHERE3:%[0-9]+]] = "ttir.where"([[WHERE0]], [[WHERE1]], [[WHERE2]], %6)
    // CHECK-SAME: -> tensor<1x1x128xf32>
    // CHECK: [[BCAST0:%[0-9]+]] = "ttir.broadcast"([[WHERE3]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<1x64x128xf32>
    // CHECK: "ttir.broadcast"([[BCAST0]], {{%[0-9]+}})
    // CHECK-SAME: -> tensor<32x64x128xf32>
    return %27 : tensor<32x64x128xf32>
  }
}
