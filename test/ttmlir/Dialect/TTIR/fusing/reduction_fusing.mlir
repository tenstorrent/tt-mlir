// RUN: ttmlir-opt --ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test basic reduction + reshape fusion (sum operation).
module {
  // CHECK-LABEL: func.func @sum_reshape_fusion
  func.func @sum_reshape_fusion(%arg0: tensor<32x64x128xf32>) -> tensor<32x1x128xf32> {
    // CHECK: %[[RESULT:.*]] = "ttir.sum"(%arg0, %{{.*}}) <{dim_arg = [1 : i32], keep_dim = true}> : (tensor<32x64x128xf32>, tensor<32x1x128xf32>) -> tensor<32x1x128xf32>
    // CHECK-NOT: ttir.reshape
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<32x128xf32>
    %1 = "ttir.sum"(%arg0, %0) {dim_arg = [1 : i32], keep_dim = false} : (tensor<32x64x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>

    %2 = ttir.empty() : tensor<32x1x128xf32>
    %3 = "ttir.reshape"(%1, %2) {shape = [32 : i32, 1 : i32, 128 : i32]} : (tensor<32x128xf32>, tensor<32x1x128xf32>) -> tensor<32x1x128xf32>

    return %3 : tensor<32x1x128xf32>
  }
}

// Test reduction + reshape fusion with mean operation.
module {
  // CHECK-LABEL: func.func @mean_reshape_fusion
  func.func @mean_reshape_fusion(%arg0: tensor<32x64x128xf32>) -> tensor<32x64x1xf32> {
    // CHECK: %[[RESULT:.*]] = "ttir.mean"(%arg0, %{{.*}}) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<32x64x128xf32>, tensor<32x64x1xf32>) -> tensor<32x64x1xf32>
    // CHECK-NOT: ttir.reshape
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<32x64xf32>
    %1 = "ttir.mean"(%arg0, %0) {dim_arg = [2 : i32], keep_dim = false} : (tensor<32x64x128xf32>, tensor<32x64xf32>) -> tensor<32x64xf32>

    %2 = ttir.empty() : tensor<32x64x1xf32>
    %3 = "ttir.reshape"(%1, %2) {shape = [32 : i32, 64 : i32, 1 : i32]} : (tensor<32x64xf32>, tensor<32x64x1xf32>) -> tensor<32x64x1xf32>

    return %3 : tensor<32x64x1xf32>
  }
}

// Test reduction + reshape fusion with max operation.
module {
  // CHECK-LABEL: func.func @max_reshape_fusion
  func.func @max_reshape_fusion(%arg0: tensor<32x64x128xf32>) -> tensor<1x64x128xf32> {
    // CHECK: %[[RESULT:.*]] = "ttir.max"(%arg0, %{{.*}}) <{dim_arg = [0 : i32], keep_dim = true}> : (tensor<32x64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>
    // CHECK-NOT: ttir.reshape
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<64x128xf32>
    %1 = "ttir.max"(%arg0, %0) {dim_arg = [0 : i32], keep_dim = false} : (tensor<32x64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>

    %2 = ttir.empty() : tensor<1x64x128xf32>
    %3 = "ttir.reshape"(%1, %2) {shape = [1 : i32, 64 : i32, 128 : i32]} : (tensor<64x128xf32>, tensor<1x64x128xf32>) -> tensor<1x64x128xf32>

    return %3 : tensor<1x64x128xf32>
  }
}

// Test reduction + reshape fusion with multiple reduction dimensions.
module {
  // CHECK-LABEL: func.func @multi_dim_reduction_reshape_fusion
  func.func @multi_dim_reduction_reshape_fusion(%arg0: tensor<32x64x128xf32>) -> tensor<32x1x1xf32> {
    // CHECK: %[[RESULT:.*]] = "ttir.sum"(%arg0, %{{.*}}) <{dim_arg = [1 : i32, 2 : i32], keep_dim = true}> : (tensor<32x64x128xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>
    // CHECK-NOT: ttir.reshape
    // CHECK: return %[[RESULT]]

    %0 = ttir.empty() : tensor<32xf32>
    %1 = "ttir.sum"(%arg0, %0) {dim_arg = [1 : i32, 2 : i32], keep_dim = false} : (tensor<32x64x128xf32>, tensor<32xf32>) -> tensor<32xf32>

    %2 = ttir.empty() : tensor<32x1x1xf32>
    %3 = "ttir.reshape"(%1, %2) {shape = [32 : i32, 1 : i32, 1 : i32]} : (tensor<32xf32>, tensor<32x1x1xf32>) -> tensor<32x1x1xf32>

    return %3 : tensor<32x1x1xf32>
  }
}

// Test negative case - reduction already has keep_dim=true.
module {
  // CHECK-LABEL: func.func @no_fusion_already_keep_dim
  func.func @no_fusion_already_keep_dim(%arg0: tensor<32x64x128xf32>) -> tensor<32x1x128xf32> {
    // CHECK: ttir.sum
    // CHECK-NOT: ttir.reshape

    // Reduction already has keep_dim=true, so we cannot fuse. Reshape will be folded.
    %0 = ttir.empty() : tensor<32x1x128xf32>
    %1 = "ttir.sum"(%arg0, %0) {dim_arg = [1 : i32], keep_dim = true} : (tensor<32x64x128xf32>, tensor<32x1x128xf32>) -> tensor<32x1x128xf32>

    %2 = ttir.empty() : tensor<32x1x128xf32>
    %3 = "ttir.reshape"(%1, %2) {shape = [32 : i32, 1 : i32, 128 : i32]} : (tensor<32x1x128xf32>, tensor<32x1x128xf32>) -> tensor<32x1x128xf32>

    return %1 : tensor<32x1x128xf32>
  }
}

// Test negative case - reduction has more than one use.
module {
  // CHECK-LABEL: func.func @no_fusion_more_than_one_use
  func.func @no_fusion_more_than_one_use(%arg0: tensor<32x64x128xf32>) -> tensor<32x1x128xf32> {
    // CHECK: ttir.sum
    // CHECK: ttir.add
    // CHECK: ttir.reshape
    // CHECK: ttir.reshape
    // CHECK: ttir.add

    %0 = ttir.empty() : tensor<32x128xf32>
    %1 = "ttir.sum"(%arg0, %0) {dim_arg = [1 : i32], keep_dim = false} : (tensor<32x64x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>

    // Extra use of the reduction result prevents fusion.
    %2 = ttir.empty() : tensor<32x128xf32>
    %3 = "ttir.add"(%1, %1, %2) : (tensor<32x128xf32>, tensor<32x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>

    %4 = ttir.empty() : tensor<32x1x128xf32>
    %5 = "ttir.reshape"(%3, %4) {shape = [32 : i32, 1 : i32, 128 : i32]} : (tensor<32x128xf32>, tensor<32x1x128xf32>) -> tensor<32x1x128xf32>

    %6 = ttir.empty() : tensor<32x1x128xf32>
    %7 = "ttir.reshape"(%1, %6) {shape = [32 : i32, 1 : i32, 128 : i32]} : (tensor<32x128xf32>, tensor<32x1x128xf32>) -> tensor<32x1x128xf32>

    %8 = ttir.empty() : tensor<32x1x128xf32>
    %9 = "ttir.add"(%5, %7, %8) : (tensor<32x1x128xf32>, tensor<32x1x128xf32>, tensor<32x1x128xf32>) -> tensor<32x1x128xf32>

    return %9 : tensor<32x1x128xf32>
  }
}

// Test negative case - cannot fuse reduction + reshape because reshape rank and reduce rank dont match.
module {
  // CHECK-LABEL: func.func @no_fusion_rank_mismatch
  func.func @no_fusion_rank_mismatch(%arg0: tensor<32x64x128xf32>) -> tensor<32x1x1x128xf32> {
    // CHECK: ttir.sum
    // CHECK: ttir.reshape

    %0 = ttir.empty() : tensor<32x128xf32>
    %1 = "ttir.sum"(%arg0, %0) {dim_arg = [1 : i32], keep_dim = false} : (tensor<32x64x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>

    // Reshape unsqueezes the result to greater rank so cannot fuse.
    %2 = ttir.empty() : tensor<32x1x1x128xf32>
    %3 = "ttir.reshape"(%1, %2) {shape = [32 : i32, 1 : i32, 1 : i32, 128 : i32]} : (tensor<32x128xf32>, tensor<32x1x1x128xf32>) -> tensor<32x1x1x128xf32>

    return %3 : tensor<32x1x1x128xf32>
  }
}

// Test negative case - cannot fuse reduction + reshape because reduced dims don't match.
module {
  // CHECK-LABEL: func.func @no_fusion_reduced_dims_mismatch
  func.func @no_fusion_reduced_dims_mismatch(%arg0: tensor<32x64x128xf32>) -> tensor<1x32x128xf32> {
    // CHECK: ttir.sum
    // CHECK: ttir.reshape

    %0 = ttir.empty() : tensor<32x128xf32>
    %1 = "ttir.sum"(%arg0, %0) {dim_arg = [1 : i32], keep_dim = false} : (tensor<32x64x128xf32>, tensor<32x128xf32>) -> tensor<32x128xf32>

    // Reshape adds new dimension but it's not the same as the reduction dim.
    %2 = ttir.empty() : tensor<1x32x128xf32>
    %3 = "ttir.reshape"(%1, %2) {shape = [1 : i32, 32 : i32, 128 : i32]} : (tensor<32x128xf32>, tensor<1x32x128xf32>) -> tensor<1x32x128xf32>

    return %3 : tensor<1x32x128xf32>
  }
}
