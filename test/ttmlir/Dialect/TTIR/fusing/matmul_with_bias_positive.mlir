// RUN: ttmlir-opt -ttir-to-ttir-decomposition -ttir-implicit-broadcast-fold -ttir-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// ===----------------------------------------------------------------------===
// POSITIVE CASES: Operations that SHOULD be fused into linear op with bias
// ===----------------------------------------------------------------------===

module {
  func.func @dot_general_with_bias_1(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<1024xf32>) -> tensor<68x1024xf32> {
    // CHECK-LABEL: func.func @dot_general_with_bias_1
    // CHECK: "ttir.linear"(%arg0, %arg1, %arg2)
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> (tensor<68x1024xf32>)
    %3 = "ttir.add"(%1, %bias) : (tensor<68x1024xf32>, tensor<1024xf32>) -> tensor<68x1024xf32>
    return %3 : tensor<68x1024xf32>
  }
}

module {
  // Replace order of operands for add op.
  func.func @dot_general_with_bias_2(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<1024xf32>) -> tensor<2x34x16x64xf32> {
    // CHECK-LABEL: func.func @dot_general_with_bias_2
    // CHECK: "ttir.linear"(%arg0, %arg1, %arg2)
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %3 = "ttir.add"(%bias, %1) : (tensor<1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
    %5 = "ttir.reshape"(%3)<{shape = [2 : i32, 34 : i32, 16 : i32, 64 : i32]}> : (tensor<68x1024xf32>) -> tensor<2x34x16x64xf32>
    return %5 : tensor<2x34x16x64xf32>
  }
}

module {
  // dot_general op followed by reshape op before add op.
  func.func @dot_general_with_bias_3(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<1024xf32>) -> tensor<2x34x1024xf32> {
    // CHECK-LABEL: func.func @dot_general_with_bias_3
    // CHECK: "ttir.linear"(%arg0, %arg1, %arg2)
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    // CHECK: "ttir.reshape"(%0)
    // CHECK-SAME: (tensor<68x1024xf32>) -> tensor<2x34x1024xf32>
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %3 = "ttir.reshape"(%1) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf32>) -> tensor<2x34x1024xf32>
    %5 = "ttir.add"(%3, %bias) : (tensor<2x34x1024xf32>, tensor<1024xf32>) -> tensor<2x34x1024xf32>
    return %5 : tensor<2x34x1024xf32>
  }
}

module {
  func.func @dot_general_with_bias_4(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<1024xf32>) -> tensor<2x34x1024xf32> {
    // CHECK-LABEL: func.func @dot_general_with_bias_4
    // CHECK: "ttir.linear"(%arg0, %arg1, %arg2)
    // CHECK: "ttir.reshape"(%0)
    // CHECK-SAME: (tensor<68x1024xf32>) -> tensor<2x34x1024xf32>
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> tensor<68x1024xf32>
    %3 = "ttir.reshape"(%1) <{shape = [2 : i32, 34 : i32, 1024 : i32]}> : (tensor<68x1024xf32>) -> tensor<2x34x1024xf32>
    %5 = "ttir.reshape"(%bias) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xf32>) -> tensor<1x1x1024xf32>
    %7 = "ttir.broadcast"(%5) <{broadcast_dimensions = array<i64: 2, 34, 1>}> : (tensor<1x1x1024xf32>) -> tensor<2x34x1024xf32>
    %9 = "ttir.add"(%3, %7) : (tensor<2x34x1024xf32>, tensor<2x34x1024xf32>) -> tensor<2x34x1024xf32>
    return %9 : tensor<2x34x1024xf32>
  }
}

module {
  func.func @dot_general_with_bias_5(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<2x68x1024xf32>) -> tensor<2x68x1024xf32> {
    // CHECK-LABEL: func.func @dot_general_with_bias_5
    // CHECK: "ttir.linear"(%arg0, %arg1, %arg2)
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> (tensor<68x1024xf32>)
    %3 = "ttir.add"(%1, %bias) : (tensor<68x1024xf32>, tensor<2x68x1024xf32>) -> tensor<2x68x1024xf32>
    return %3 : tensor<2x68x1024xf32>
  }
}

module {
  func.func @dot_general_with_bias_6(%arg0: tensor<1576x2xf32>, %arg1: tensor<2x768xf32>, %bias: tensor<768xf32>) -> tensor<8x197x768xf32> {
    // CHECK-LABEL: func.func @dot_general_with_bias_6
    // CHECK: "ttir.linear"(%arg0, %arg1, %arg2)
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    // CHECK: "ttir.reshape"(%0)
    // CHECK-SAME: (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<1576x2xf32>, tensor<2x768xf32>) -> tensor<1576x768xf32>
    %2 = "ttir.reshape"(%0) <{shape = [8 : i32, 197 : i32, 768 : i32]}> : (tensor<1576x768xf32>) -> tensor<8x197x768xf32>
    %4 = "ttir.add"(%2, %bias) : (tensor<8x197x768xf32>, tensor<768xf32>) -> tensor<8x197x768xf32>
    return %4 : tensor<8x197x768xf32>
  }
}

module {
  func.func @dot_general_with_bias_7(%arg0: tensor<256x1024xbf16>, %arg1: tensor<1024x1024xbf16>, %bias: tensor<1024xbf16>) -> tensor<1x256x1024xbf16> {
    // CHECK-LABEL: func.func @dot_general_with_bias_7
    // CHECK: "ttir.linear"(%arg0, %arg1, %arg2)
    // CHECK: "ttir.reshape"(%0)
    // CHECK-SAME: (tensor<256x1024xbf16>) -> tensor<1x256x1024xbf16>
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<256x1024xbf16>, tensor<1024x1024xbf16>) -> tensor<256x1024xbf16>
    %2 = "ttir.reshape"(%0) <{shape = [1 : i32, 256 : i32, 1024 : i32]}> : (tensor<256x1024xbf16>) -> tensor<1x256x1024xbf16>
    %4 = "ttir.reshape"(%bias) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %6 = "ttir.reshape"(%4) <{shape = [1024 : i32]}> : (tensor<1x1x1024xbf16>) -> tensor<1024xbf16>
    %8 = "ttir.reshape"(%6) <{shape = [1 : i32, 1 : i32, 1024 : i32]}> : (tensor<1024xbf16>) -> tensor<1x1x1024xbf16>
    %10 = "ttir.broadcast"(%8) <{broadcast_dimensions = array<i64: 1, 256, 1>}> : (tensor<1x1x1024xbf16>) -> tensor<1x256x1024xbf16>
    %12 = "ttir.add"(%2, %10) : (tensor<1x256x1024xbf16>, tensor<1x256x1024xbf16>) -> tensor<1x256x1024xbf16>
    return %12 : tensor<1x256x1024xbf16>
  }
}

module {
  func.func @dot_general_with_bias_8(%arg0: tensor<16x256x64xbf16>, %arg1: tensor<16x64x256xbf16>, %bias: tensor<1x256xbf16>) -> tensor<1x16x256x256xbf16> {
    // CHECK-LABEL: func.func @dot_general_with_bias_8
    // CHECK: "ttir.linear"(%arg0, %arg1, %arg2)
    // CHECK: "ttir.reshape"(%0)
    // CHECK-SAME: (tensor<16x256x256xbf16>) -> tensor<1x16x256x256xbf16>
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64: 0>, batch_dims_rhs = array<i64: 0>, contract_dims_lhs = array<i64: 2>, contract_dims_rhs = array<i64: 1>}> : (tensor<16x256x64xbf16>, tensor<16x64x256xbf16>) -> tensor<16x256x256xbf16>
    %2 = "ttir.reshape"(%0) <{shape = [1 : i32, 16 : i32, 256 : i32, 256 : i32]}> : (tensor<16x256x256xbf16>) -> tensor<1x16x256x256xbf16>
    %4 = "ttir.reshape"(%bias) <{shape = [1 : i32, 1 : i32, 256 : i32]}> : (tensor<1x256xbf16>) -> tensor<1x1x256xbf16>
    %6 = "ttir.reshape"(%4) <{shape = [1 : i32, 1 : i32, 1 : i32, 256 : i32]}> : (tensor<1x1x256xbf16>) -> tensor<1x1x1x256xbf16>
    %8 = "ttir.broadcast"(%6) <{broadcast_dimensions = array<i64: 1, 1, 256, 1>}> : (tensor<1x1x1x256xbf16>) -> tensor<1x1x256x256xbf16>
    %10 = "ttir.broadcast"(%8) <{broadcast_dimensions = array<i64: 1, 16, 1, 1>}> : (tensor<1x1x256x256xbf16>) -> tensor<1x16x256x256xbf16>
    %12 = "ttir.add"(%2, %10) : (tensor<1x16x256x256xbf16>, tensor<1x16x256x256xbf16>) -> tensor<1x16x256x256xbf16>
    return %12 : tensor<1x16x256x256xbf16>
  }
}

module {
  func.func @dot_general_with_bias_9(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<1024xf32>) -> tensor<68x1024xf32> {
    // CHECK-LABEL: func.func @dot_general_with_bias_9
    // CHECK: "ttir.linear"(%arg0, %arg1, %arg2)
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> (tensor<68x1024xf32>)
    %3 = "ttir.reshape" (%bias) <{shape = [1 : i32, 1024 : i32]}> : (tensor<1024xf32>) -> tensor<1x1024xf32>
    %5 = "ttir.broadcast"(%3) <{broadcast_dimensions = array<i64: 68, 1>}> : (tensor<1x1024xf32>) -> tensor<68x1024xf32>
    %7 = "ttir.add"(%1, %5) : (tensor<68x1024xf32>, tensor<68x1024xf32>) -> tensor<68x1024xf32>
    return %7 : tensor<68x1024xf32>
  }
}

module {
  func.func @dot_general_with_bias_10(%arg0: tensor<68x1024xf32>, %arg1: tensor<1024x1024xf32>, %bias: tensor<2x512xf32>) -> tensor<1x68x1024xf32> {
    // CHECK-LABEL: func.func @dot_general_with_bias_10
    // CHECK: "ttir.reshape"(%arg2)
    // CHECK-SAME: (tensor<2x512xf32>) -> tensor<1024xf32>
    // CHECK: "ttir.linear"(%arg0, %arg1, %0)
    // CHECK: "ttir.reshape"(%1)
    // CHECK-SAME: (tensor<68x1024xf32>) -> tensor<1x68x1024xf32>
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.reshape"(%bias) <{shape = [1024 : i32]}> : (tensor<2x512xf32>) -> tensor<1024xf32>
    %1 = "ttir.dot_general"(%arg0, %arg1) <{batch_dims_lhs = array<i64>, batch_dims_rhs = array<i64>, contract_dims_lhs = array<i64: 1>, contract_dims_rhs = array<i64: 0>}> : (tensor<68x1024xf32>, tensor<1024x1024xf32>) -> (tensor<68x1024xf32>)
    %2 = "ttir.reshape"(%1) <{shape = [1 : i32, 68 : i32, 1024 : i32]}> : (tensor<68x1024xf32>) -> tensor<1x68x1024xf32>

    %3 = "ttir.add"(%2, %0) : (tensor<1x68x1024xf32>, tensor<1024xf32>) -> tensor<1x68x1024xf32>
    return %3 : tensor<1x68x1024xf32>
  }
}

module {
  func.func @dot_general_with_bias_11(%arg0: tensor<32x2048xbf16>, %arg1: tensor<2048x51200xbf16>, %bias: tensor<32x1x51200xbf16>) -> tensor<32x1x51200xbf16> {
    // CHECK-LABEL: func.func @dot_general_with_bias_11
    // CHECK: "ttir.reshape"(%arg2)
    // CHECK-SAME: (tensor<32x1x51200xbf16>) -> tensor<32x51200xbf16>
    // CHECK: "ttir.linear"(%arg0, %arg1, %0)
    // CHECK: "ttir.reshape"(%1)
    // CHECK-SAME: (tensor<32x51200xbf16>) -> tensor<32x1x51200xbf16>
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<32x2048xbf16>, tensor<2048x51200xbf16>) -> tensor<32x51200xbf16>
    %1 = "ttir.reshape"(%bias) <{shape = [32 : i32, 51200 : i32]}> : (tensor<32x1x51200xbf16>) -> tensor<32x51200xbf16>
    %2 = "ttir.add"(%0, %1) : (tensor<32x51200xbf16>, tensor<32x51200xbf16>) -> tensor<32x51200xbf16>
    %3 = "ttir.reshape"(%2) <{shape = [32 : i32, 1 : i32, 51200 : i32]}> : (tensor<32x51200xbf16>) -> tensor<32x1x51200xbf16>
    return %3 : tensor<32x1x51200xbf16>
  }
}

module {
  func.func @dot_general_with_bias_12(%arg0: tensor<1x1x68x2048xbf16>, %arg1: tensor<2048x51200xbf16>, %bias: tensor<1x68x51200xbf16>) -> tensor<1x1x68x51200xbf16> {
    // CHECK-LABEL: func.func @dot_general_with_bias_12
    // CHECK: "ttir.linear"(%arg0, %arg1, %arg2)
    // CHECK-SAME: (tensor<1x1x68x2048xbf16>, tensor<2048x51200xbf16>, tensor<1x68x51200xbf16>) -> tensor<1x1x68x51200xbf16>
    // CHECK-NOT: "ttir.dot_general"
    // CHECK-NOT: "ttir.matmul"
    // CHECK-NOT: "ttir.add"
    %0 = "ttir.matmul"(%arg0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x1x68x2048xbf16>, tensor<2048x51200xbf16>) -> tensor<1x1x68x51200xbf16>
    %1 = "ttir.add"(%0, %bias) : (tensor<1x1x68x51200xbf16>, tensor<1x68x51200xbf16>) -> tensor<1x1x68x51200xbf16>
    return %1 : tensor<1x1x68x51200xbf16>
  }
}
