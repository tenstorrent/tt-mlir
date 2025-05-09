// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @implicit_broadcast_both_operands_maximum(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.maximum"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.maximum"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_maximum(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.maximum"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.maximum"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_maximum(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.maximum"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.maximum"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_minimum(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.minimum"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.minimum"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_minimum(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.minimum"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.minimum"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_minimum(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.minimum"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.minimum"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_pow(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.pow"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.pow"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_pow(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.pow"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.pow"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_pow(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.pow"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.pow"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_bitwise_and(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.bitwise_and"
  // CHECK-SAME: tensor<1x16x1xsi32
  // CHECK-SAME: tensor<1x1x32xsi32
  // CHECK-SAME: -> tensor<1x16x32xsi32
  %1 = "ttir.bitwise_and"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_bitwise_and(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.bitwise_and"
  // CHECK-SAME: tensor<1x16x32xsi32
  // CHECK-SAME: tensor<8x16x32xsi32
  // CHECK-SAME: -> tensor<8x16x32xsi32
  %1 = "ttir.bitwise_and"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_bitwise_and(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.bitwise_and"
  // CHECK-SAME: tensor<8x16x32xsi32
  // CHECK-SAME: tensor<1x16x32xsi32
  // CHECK-SAME: -> tensor<8x16x32xsi32
  %1 = "ttir.bitwise_and"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_bitwise_or(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.bitwise_or"
  // CHECK-SAME: tensor<1x16x1xsi32
  // CHECK-SAME: tensor<1x1x32xsi32
  // CHECK-SAME: -> tensor<1x16x32xsi32
  %1 = "ttir.bitwise_or"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_bitwise_or(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.bitwise_or"
  // CHECK-SAME: tensor<1x16x32xsi32
  // CHECK-SAME: tensor<8x16x32xsi32
  // CHECK-SAME: -> tensor<8x16x32xsi32
  %1 = "ttir.bitwise_or"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_bitwise_or(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.bitwise_or"
  // CHECK-SAME: tensor<8x16x32xsi32
  // CHECK-SAME: tensor<1x16x32xsi32
  // CHECK-SAME: -> tensor<8x16x32xsi32
  %1 = "ttir.bitwise_or"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_bitwise_xor(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.bitwise_xor"
  // CHECK-SAME: tensor<1x16x1xsi32
  // CHECK-SAME: tensor<1x1x32xsi32
  // CHECK-SAME: -> tensor<1x16x32xsi32
  %1 = "ttir.bitwise_xor"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_bitwise_xor(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.bitwise_xor"
  // CHECK-SAME: tensor<1x16x32xsi32
  // CHECK-SAME: tensor<8x16x32xsi32
  // CHECK-SAME: -> tensor<8x16x32xsi32
  %1 = "ttir.bitwise_xor"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_bitwise_xor(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.bitwise_xor"
  // CHECK-SAME: tensor<8x16x32xsi32
  // CHECK-SAME: tensor<1x16x32xsi32
  // CHECK-SAME: -> tensor<8x16x32xsi32
  %1 = "ttir.bitwise_xor"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}
