// RUN: ttmlir-opt --ttir-implicit-broadcast-fold --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // CHECK-LABEL: func.func @softmax_fusion_dim1
  func.func @softmax_fusion_dim1(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK-NOT: ttir.exp
    // CHECK-NOT: ttir.sum
    // CHECK-NOT: ttir.broadcast
    // CHECK-NOT: ttir.div
    // CHECK: %[[RESULT:.*]] = "ttir.softmax"(%arg0, %{{.*}}) <{dimension = 1 : si32, numericStable = false}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = "ttir.exp"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %2 = ttir.empty() : tensor<32x1xf32>
    %3 = "ttir.sum"(%1, %2) {dim_arg = [1 : i32], keep_dim = true} : (tensor<32x32xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>

    %4 = ttir.empty() : tensor<32x32xf32>
    %5 = "ttir.broadcast"(%3, %4) {broadcast_dimensions = array<i64: 1, 32>} : (tensor<32x1xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %6 = ttir.empty() : tensor<32x32xf32>
    %7 = "ttir.div"(%1, %5, %6) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %7 : tensor<32x32xf32>
  }
}

module {
  // CHECK-LABEL: func.func @softmax_fusion_dim0
  func.func @softmax_fusion_dim0(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK-NOT: ttir.exp
    // CHECK-NOT: ttir.sum
    // CHECK-NOT: ttir.broadcast
    // CHECK-NOT: ttir.div
    // CHECK: %[[RESULT:.*]] = "ttir.softmax"(%arg0, %{{.*}}) <{dimension = 0 : si32, numericStable = false}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = "ttir.exp"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %2 = ttir.empty() : tensor<1x32xf32>
    %3 = "ttir.sum"(%1, %2) {dim_arg = [0 : i32], keep_dim = true} : (tensor<32x32xf32>, tensor<1x32xf32>) -> tensor<1x32xf32>

    %4 = ttir.empty() : tensor<32x32xf32>
    %5 = "ttir.broadcast"(%3, %4) {broadcast_dimensions = array<i64: 32, 1>} : (tensor<1x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %6 = ttir.empty() : tensor<32x32xf32>
    %7 = "ttir.div"(%1, %5, %6) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %7 : tensor<32x32xf32>
  }
}

// Test case with a negative pattern - different input to exp and sum.
module {
  // CHECK-LABEL: func.func @no_fusion_different_inputs
  func.func @no_fusion_different_inputs(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK: ttir.exp
    // CHECK: ttir.sum
    // CHECK-NOT: ttir.broadcast
    // CHECK: ttir.div
    // CHECK-NOT: ttir.softmax

    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = "ttir.exp"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // Here we are using %%arg1 for reduction instead of the exp output, so we cannot fuse.
    %2 = ttir.empty() : tensor<32x1xf32>
    %3 = "ttir.sum"(%arg1, %2) {dim_arg = [1 : i32], keep_dim = true} : (tensor<32x32xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>

    %4 = ttir.empty() : tensor<32x32xf32>
    %5 = "ttir.broadcast"(%3, %4) {broadcast_dimensions = array<i64: 1, 32>} : (tensor<32x1xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %6 = ttir.empty() : tensor<32x32xf32>
    %7 = "ttir.div"(%1, %5, %6) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %7 : tensor<32x32xf32>
  }
}

// Test case where rehape is fused into reduction and we can fuse into softmax.
module {
  // CHECK-LABEL: func.func @no_fusion_keep_dim_false
  func.func @no_fusion_keep_dim_false(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK-NOT: ttir.exp
    // CHECK-NOT: ttir.sum
    // CHECK-NOT: ttir.broadcast
    // CHECK-NOT: ttir.div
    // CHECK: %[[RESULT:.*]] = "ttir.softmax"(%arg0, %{{.*}}) <{dimension = 1 : si32, numericStable = false}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = "ttir.exp"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %2 = ttir.empty() : tensor<32xf32>
    %3 = "ttir.sum"(%1, %2) {dim_arg = [1 : i32], keep_dim = false} : (tensor<32x32xf32>, tensor<32xf32>) -> tensor<32xf32>

    %4 = ttir.empty() : tensor<32x1xf32>
    %5 = "ttir.reshape"(%3, %4) {shape = [32 : i32, 1 : i32]} : (tensor<32xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>

    %6 = ttir.empty() : tensor<32x32xf32>
    %7 = "ttir.broadcast"(%5, %6) {broadcast_dimensions = array<i64: 1, 32>} : (tensor<32x1xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %8 = ttir.empty() : tensor<32x32xf32>
    %9 = "ttir.div"(%1, %7, %8) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %9 : tensor<32x32xf32>
  }
}

// Test case with a negative pattern - exp has more than two users.
module {
  // CHECK-LABEL: func.func @no_fusion_exp_multiple_users
  func.func @no_fusion_exp_multiple_users(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK: ttir.exp
    // CHECK: ttir.sum
    // CHECK-NOT: ttir.broadcast
    // CHECK: ttir.div
    // CHECK: ttir.add
    // CHECK-NOT: ttir.softmax

    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = "ttir.exp"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %2 = ttir.empty() : tensor<32x1xf32>
    %3 = "ttir.sum"(%1, %2) {dim_arg = [1 : i32], keep_dim = true} : (tensor<32x32xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>

    %4 = ttir.empty() : tensor<32x32xf32>
    %5 = "ttir.broadcast"(%3, %4) {broadcast_dimensions = array<i64: 1, 32>} : (tensor<32x1xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %6 = ttir.empty() : tensor<32x32xf32>
    %7 = "ttir.div"(%1, %5, %6) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // Extra use of the exp result prevents fusion.
    %8 = ttir.empty() : tensor<32x32xf32>
    %9 = "ttir.add"(%1, %7, %8) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %9 : tensor<32x32xf32>
  }
}

// Test case with a negative pattern - sum has more than one user.
module {
  // CHECK-LABEL: func.func @no_fusion_sum_multiple_users
  func.func @no_fusion_sum_multiple_users(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK: ttir.exp
    // CHECK: ttir.sum
    // CHECK-NOT: ttir.broadcast
    // CHECK: ttir.div
    // CHECK: ttir.add
    // CHECK-NOT: ttir.softmax

    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = "ttir.exp"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %2 = ttir.empty() : tensor<32x1xf32>
    %3 = "ttir.sum"(%1, %2) {dim_arg = [1 : i32], keep_dim = true} : (tensor<32x32xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>

    %4 = ttir.empty() : tensor<32x32xf32>
    %5 = "ttir.broadcast"(%3, %4) {broadcast_dimensions = array<i64: 1, 32>} : (tensor<32x1xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %6 = ttir.empty() : tensor<32x32xf32>
    %7 = "ttir.div"(%1, %5, %6) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // Extra use of the sum result prevents fusion.
    %8 = ttir.empty() : tensor<32x32xf32>
    %9 = "ttir.add"(%3, %7, %8) : (tensor<32x1xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %9: tensor<32x32xf32>
  }
}

// Test case with a negative pattern - broadcast has more than one user.
module {
  // CHECK-LABEL: func.func @no_fusion_broadcast_multiple_users
  func.func @no_fusion_broadcast_multiple_users(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK: ttir.exp
    // CHECK: ttir.sum
    // CHECK-NOT: ttir.broadcast
    // CHECK: ttir.div
    // CHECK: ttir.add
    // CHECK-NOT: ttir.softmax

    %0 = ttir.empty() : tensor<32x32xf32>
    %1 = "ttir.exp"(%arg0, %0) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %2 = ttir.empty() : tensor<32x1xf32>
    %3 = "ttir.sum"(%1, %2) {dim_arg = [1 : i32], keep_dim = true} : (tensor<32x32xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>

    %4 = ttir.empty() : tensor<32x32xf32>
    %5 = "ttir.broadcast"(%3, %4) {broadcast_dimensions = array<i64: 1, 32>} : (tensor<32x1xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %6 = ttir.empty() : tensor<32x32xf32>
    %7 = "ttir.div"(%1, %5, %6) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    // Extra use of the broadcast result prevents fusion.
    %8 = ttir.empty() : tensor<32x32xf32>
    %9 = "ttir.add"(%5, %7, %8) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %9 : tensor<32x32xf32>
  }
}

// Test case for fusing numerically stable softmax pattern.
module {
  // CHECK-LABEL: func.func @softmax_numeric_stable_fusion
  func.func @softmax_numeric_stable_fusion(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // After first fusion pass: div(exp(x-max(x)), sum(exp(x-max(x)))) -> softmax(x-max(x))
    // After second fusion pass: softmax(x-max(x)) -> softmax(x, numericStable=true)
    // CHECK-NOT: ttir.max
    // CHECK-NOT: ttir.subtract
    // CHECK: %[[RESULT:.*]] = "ttir.softmax"(%arg0, %{{.*}}) <{dimension = 1 : si32, numericStable = true}> : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<32x1xf32>
    %1 = "ttir.max"(%arg0, %0) {dim_arg = [1 : i32], keep_dim = true} : (tensor<32x32xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>

    %2 = ttir.empty() : tensor<32x32xf32>
    %3 = "ttir.broadcast"(%1, %2) {broadcast_dimensions = array<i64: 1, 32>} : (tensor<32x1xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %4 = ttir.empty() : tensor<32x32xf32>
    %5 = "ttir.subtract"(%arg0, %3, %4) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %6 = ttir.empty() : tensor<32x32xf32>
    %7 = "ttir.exp"(%5, %6) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %8 = ttir.empty() : tensor<32x1xf32>
    %9 = "ttir.sum"(%7, %8) {dim_arg = [1 : i32], keep_dim = true} : (tensor<32x32xf32>, tensor<32x1xf32>) -> tensor<32x1xf32>

    %10 = ttir.empty() : tensor<32x32xf32>
    %11 = "ttir.broadcast"(%9, %10) {broadcast_dimensions = array<i64: 1, 32>} : (tensor<32x1xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    %12 = ttir.empty() : tensor<32x32xf32>
    %13 = "ttir.div"(%7, %11, %12) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>

    return %13 : tensor<32x32xf32>
  }
}
