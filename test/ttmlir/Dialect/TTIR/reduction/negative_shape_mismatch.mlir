// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s
// Negative tests for reduce ops expected shape mismatch.

module {
  // CHECK: error: 'ttir.sum' op Expected output shape (128, 16), got (128, 16, 1)
  func.func public @shape_mismatch_0(%arg0: tensor<128x16x32xf32>) -> tensor<128x16x1xf32> {
    %0 = ttir.empty() : tensor<128x16x1xf32>
    %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [2 : i32], keep_dim = false}> : (tensor<128x16x32xf32>, tensor<128x16x1xf32>) -> tensor<128x16x1xf32>
    return %1 : tensor<128x16x1xf32>
  }
}

// -----
module {
  // CHECK: error: 'ttir.sum' op Expected output shape (128, 16, 1), got (128, 16)
  func.func public @shape_mismatch_1(%arg0: tensor<128x16x32xf32>) -> tensor<128x16xf32> {
    %0 = ttir.empty() : tensor<128x16xf32>
    %1 = "ttir.sum"(%arg0, %0) <{dim_arg = [2 : i32], keep_dim = true}> : (tensor<128x16x32xf32>, tensor<128x16xf32>) -> tensor<128x16xf32>
    return %1 : tensor<128x16xf32>
  }
}

// -----
module {
  // CHECK: error: 'ttir.sum' op Expected output shape (1, 1, 1), got (128, 16, 1)
  func.func public @shape_mismatch_2(%arg0: tensor<128x16x32xf32>) -> tensor<128x16x1xf32> {
    %0 = ttir.empty() : tensor<128x16x1xf32>
    %1 = "ttir.sum"(%arg0, %0) <{keep_dim = true}> : (tensor<128x16x32xf32>, tensor<128x16x1xf32>) -> tensor<128x16x1xf32>
    return %1 : tensor<128x16x1xf32>
  }
}

// -----
module {
  // CHECK: error: 'ttir.sum' op Expected output shape (), got (128, 16, 1)
  func.func public @shape_mismatch_3(%arg0: tensor<128x16x32xf32>) -> tensor<128x16x1xf32> {
    %0 = ttir.empty() : tensor<128x16x1xf32>
    %1 = "ttir.sum"(%arg0, %0) <{keep_dim = false}> : (tensor<128x16x32xf32>, tensor<128x16x1xf32>) -> tensor<128x16x1xf32>
    return %1 : tensor<128x16x1xf32>
  }
}
