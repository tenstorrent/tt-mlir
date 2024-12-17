// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s  > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
module attributes {} {
  func.func @neg_dim_five(%arg0: tensor<4x2x32x32xbf16>) -> tensor<1x4x2x32x32xbf16> {
    %0 = tensor.empty() : tensor<1x4x2x32x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 4 : i32, 2 : i32, 32 : i32, 32 : i32]}
    // CHECK-SAME: tensor<4x2x32x32xbf16,
    // CHECK-SAME: -> tensor<1x4x2x32x32xbf16
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = -5 : si32}> : (tensor<4x2x32x32xbf16>, tensor<1x4x2x32x32xbf16>) -> tensor<1x4x2x32x32xbf16>
    return %1 : tensor<1x4x2x32x32xbf16>
  }
}

module attributes {} {
  func.func @neg_dim_four(%arg0: tensor<4x2x32x32xbf16>) -> tensor<4x1x2x32x32xbf16> {
    %0 = tensor.empty() : tensor<4x1x2x32x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"
    // CHECK-SAME: shape = [4 : i32, 1 : i32, 2 : i32, 32 : i32, 32 : i32]
    // CHECK-SAME: tensor<4x2x32x32xbf16,
    // CHECK-SAME: -> tensor<4x1x2x32x32xbf16
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = -4 : si32}> : (tensor<4x2x32x32xbf16>, tensor<4x1x2x32x32xbf16>) -> tensor<4x1x2x32x32xbf16>
    return %1 : tensor<4x1x2x32x32xbf16>
  }
}

module attributes {} {
  func.func @neg_dim_three(%arg0: tensor<4x2x32x32xbf16>) -> tensor<4x2x1x32x32xbf16> {
    %0 = tensor.empty() : tensor<4x2x1x32x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"
    // CHECK-SAME: shape = [4 : i32, 2 : i32, 1 : i32, 32 : i32, 32 : i32]
    // CHECK-SAME: tensor<4x2x32x32xbf16,
    // CHECK-SAME: -> tensor<4x2x1x32x32xbf16
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = -3 : si32}> : (tensor<4x2x32x32xbf16>, tensor<4x2x1x32x32xbf16>) -> tensor<4x2x1x32x32xbf16>
    return %1 : tensor<4x2x1x32x32xbf16>
  }
}

module attributes {} {
  func.func @neg_dim_two(%arg0: tensor<4x2x32x32xbf16>) -> tensor<4x2x32x1x32xbf16> {
    %0 = tensor.empty() : tensor<4x2x32x1x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"
    // CHECK-SAME: shape = [4 : i32, 2 : i32, 32 : i32, 1 : i32, 32 : i32]
    // CHECK-SAME: tensor<4x2x32x32xbf16,
    // CHECK-SAME: -> tensor<4x2x32x1x32xbf16
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = -2 : si32}> : (tensor<4x2x32x32xbf16>, tensor<4x2x32x1x32xbf16>) -> tensor<4x2x32x1x32xbf16>
    return %1 : tensor<4x2x32x1x32xbf16>
  }
}

module attributes {} {
  func.func @neg_dim_one(%arg0: tensor<4x2x32x32xbf16>) -> tensor<4x2x32x32x1xbf16> {
    %0 = tensor.empty() : tensor<4x2x32x32x1xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"
    // CHECK-SAME: shape = [4 : i32, 2 : i32, 32 : i32, 32 : i32, 1 : i32]
    // CHECK-SAME: tensor<4x2x32x32xbf16,
    // CHECK-SAME: -> tensor<4x2x32x32x1xbf16
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = -1 : si32}> : (tensor<4x2x32x32xbf16>, tensor<4x2x32x32x1xbf16>) -> tensor<4x2x32x32x1xbf16>
    return %1 : tensor<4x2x32x32x1xbf16>
  }
}

module attributes {} {
  func.func @pos_dim_zero(%arg0: tensor<4x2x32x32xbf16>) -> tensor<1x4x2x32x32xbf16> {
    %0 = tensor.empty() : tensor<1x4x2x32x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"
    // CHECK-SAME: shape = [1 : i32, 4 : i32, 2 : i32, 32 : i32, 32 : i32]
    // CHECK-SAME: tensor<4x2x32x32xbf16,
    // CHECK-SAME: -> tensor<1x4x2x32x32xbf16
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = 0 : si32}> : (tensor<4x2x32x32xbf16>, tensor<1x4x2x32x32xbf16>) -> tensor<1x4x2x32x32xbf16>
    return %1 : tensor<1x4x2x32x32xbf16>
  }
}

module attributes {} {
  func.func @pos_dim_one(%arg0: tensor<4x2x32x32xbf16>) -> tensor<4x1x2x32x32xbf16> {
    %0 = tensor.empty() : tensor<4x1x2x32x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"
    // CHECK-SAME: shape = [4 : i32, 1 : i32, 2 : i32, 32 : i32, 32 : i32]
    // CHECK-SAME: tensor<4x2x32x32xbf16,
    // CHECK-SAME: -> tensor<4x1x2x32x32xbf16
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = 1 : si32}> : (tensor<4x2x32x32xbf16>, tensor<4x1x2x32x32xbf16>) -> tensor<4x1x2x32x32xbf16>
    return %1 : tensor<4x1x2x32x32xbf16>
  }
}

module attributes {} {
  func.func @pos_dim_two(%arg0: tensor<4x2x32x32xbf16>) -> tensor<4x2x1x32x32xbf16> {
    %0 = tensor.empty() : tensor<4x2x1x32x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"
    // CHECK-SAME: shape = [4 : i32, 2 : i32, 1 : i32, 32 : i32, 32 : i32]
    // CHECK-SAME: tensor<4x2x32x32xbf16,
    // CHECK-SAME: -> tensor<4x2x1x32x32xbf16
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = 2 : si32}> : (tensor<4x2x32x32xbf16>, tensor<4x2x1x32x32xbf16>) -> tensor<4x2x1x32x32xbf16>
    return %1 : tensor<4x2x1x32x32xbf16>
  }
}

module attributes {} {
  func.func @pos_dim_three(%arg0: tensor<4x2x32x32xbf16>) -> tensor<4x2x32x1x32xbf16> {
    %0 = tensor.empty() : tensor<4x2x32x1x32xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"
    // CHECK-SAME: shape = [4 : i32, 2 : i32, 32 : i32, 1 : i32, 32 : i32]
    // CHECK-SAME: tensor<4x2x32x32xbf16,
    // CHECK-SAME: -> tensor<4x2x32x1x32xbf16
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = 3 : si32}> : (tensor<4x2x32x32xbf16>, tensor<4x2x32x1x32xbf16>) -> tensor<4x2x32x1x32xbf16>
    return %1 : tensor<4x2x32x1x32xbf16>
  }
}

module attributes {} {
  func.func @pos_dim_four(%arg0: tensor<4x2x32x32xbf16>) -> tensor<4x2x32x32x1xbf16> {
    %0 = tensor.empty() : tensor<4x2x32x32x1xbf16>
    // CHECK: %[[C:.*]] = "ttnn.reshape"
    // CHECK-SAME: shape = [4 : i32, 2 : i32, 32 : i32, 32 : i32, 1 : i32]
    // CHECK-SAME: tensor<4x2x32x32xbf16,
    // CHECK-SAME: -> tensor<4x2x32x32x1xbf16
    %1 = "ttir.unsqueeze"(%arg0, %0) <{dim = 4 : si32}> : (tensor<4x2x32x32xbf16>, tensor<4x2x32x32x1xbf16>) -> tensor<4x2x32x32x1xbf16>
    return %1 : tensor<4x2x32x32x1xbf16>
  }
}
