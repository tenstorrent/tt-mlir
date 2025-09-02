// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% enable-fusing-pass=true" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir


module {
  // CHECK-LABEL: func.func @softmax_fusion_dim1
  func.func @softmax_fusion_dim1(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK-NOT: ttir.exp
    // CHECK-NOT: ttir.sum
    // CHECK-NOT: ttir.broadcast
    // CHECK-NOT: ttir.div
    // CHECK: %[[RESULT:.*]] = "ttnn.softmax"(%arg0) <{dimension = 1 : si32, numericStable = false}>
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

  func.func @softmax_fusion_dim0(%arg0: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    // CHECK-NOT: ttir.exp
    // CHECK-NOT: ttir.sum
    // CHECK-NOT: ttir.broadcast
    // CHECK-NOT: ttir.div
    // CHECK: %[[RESULT:.*]] = "ttnn.softmax"(%arg0) <{dimension = 0 : si32, numericStable = false}>
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<32x32xbf16>
    %1 = "ttir.exp"(%arg0, %0) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    %2 = ttir.empty() : tensor<1x32xbf16>
    %3 = "ttir.sum"(%1, %2) {dim_arg = [0 : i32], keep_dim = true} : (tensor<32x32xbf16>, tensor<1x32xbf16>) -> tensor<1x32xbf16>

    %4 = ttir.empty() : tensor<32x32xbf16>
    %5 = "ttir.broadcast"(%3, %4) {broadcast_dimensions = array<i64: 32, 1>} : (tensor<1x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    %6 = ttir.empty() : tensor<32x32xbf16>
    %7 = "ttir.div"(%1, %5, %6) : (tensor<32x32xbf16>, tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>

    return %7 : tensor<32x32xbf16>
  }

  // Test case where rehape is fused into reduction and we can fuse into softmax.
  func.func @no_fusion_keep_dim_false(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK-NOT: ttir.exp
    // CHECK-NOT: ttir.sum
    // CHECK-NOT: ttir.broadcast
    // CHECK-NOT: ttir.div
    // CHECK: %[[RESULT:.*]] = "ttnn.softmax"(%arg0) <{dimension = 1 : si32, numericStable = false}>
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

  // Test case for numerically stable softmax fusion.
  func.func @softmax_numeric_stable_fusion(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
    // CHECK-NOT: ttir.max
    // CHECK-NOT: ttir.subtract
    // CHECK-NOT: ttir.exp
    // CHECK-NOT: ttir.sum
    // CHECK-NOT: ttir.broadcast
    // CHECK-NOT: ttir.div
    // CHECK: %[[RESULT:.*]] = "ttnn.softmax"(%arg0) <{dimension = 1 : si32, numericStable = true}>
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
