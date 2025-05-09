// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

func.func @implicit_broadcast_both_operands_add(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.add"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_add(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.add"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_add(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.add"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_multiply(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.multiply"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_multiply(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.multiply"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_multiply(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.multiply"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.multiply"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_subtract(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.add"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.subtract"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_subtract(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.add"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.subtract"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_subtract(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.add"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.subtract"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_equal(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.eq"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.eq"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_equal(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.eq"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.eq"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_equal(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.eq"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.eq"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_not_equal(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.ne"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.ne"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_not_equal(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.ne"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.ne"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_not_equal(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.ne"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.ne"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_greater_equal(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.ge"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.ge"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_greater_equal(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.ge"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.ge"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_greater_equal(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.ge"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.ge"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_greater_than(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.gt"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.gt"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_greater_than(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.gt"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.gt"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_greater_than(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.gt"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.gt"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_less_equal(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.le"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.le"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_less_equal(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.le"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.le"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_less_equal(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.le"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.le"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_less_than(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.lt"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.lt"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_less_than(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.lt"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.lt"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_less_than(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.lt"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.lt"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_divide(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.divide"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.div"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_divide(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.divide"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.div"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_divide(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.divide"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.div"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_logical_and(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.logical_and"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.logical_and"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_logical_and(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.logical_and"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.logical_and"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_logical_and(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.logical_and"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.logical_and"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_logical_or(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.logical_or"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.logical_or"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_logical_or(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.logical_or"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.logical_or"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_logical_or(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.logical_or"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.logical_or"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_both_operands_logical_xor(%arg0: tensor<1x16x1xf32>, %arg1: tensor<1x1x32xf32>) -> tensor<1x16x32xf32> {
  %0 = ttir.empty() : tensor<1x16x32xf32>
  // CHECK: "ttnn.logical_xor"
  // CHECK-SAME: tensor<1x16x1xf32
  // CHECK-SAME: tensor<1x1x32xf32
  // CHECK-SAME: -> tensor<1x16x32xf32
  %1 = "ttir.logical_xor"(%arg0, %arg1, %0) : (tensor<1x16x1xf32>, tensor<1x1x32xf32>, tensor<1x16x32xf32>) -> tensor<1x16x32xf32>
  return %1 : tensor<1x16x32xf32>
}

func.func @implicit_broadcast_first_operand_logical_xor(%arg0: tensor<1x16x32xf32>, %arg1: tensor<8x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.logical_xor"
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.logical_xor"(%arg0, %arg1, %0) : (tensor<1x16x32xf32>, tensor<8x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}

func.func @implicit_broadcast_second_operand_logical_xor(%arg0: tensor<8x16x32xf32>, %arg1: tensor<1x16x32xf32>) -> tensor<8x16x32xf32> {
  %0 = ttir.empty() : tensor<8x16x32xf32>
  // CHECK: "ttnn.logical_xor"
  // CHECK-SAME: tensor<8x16x32xf32
  // CHECK-SAME: tensor<1x16x32xf32
  // CHECK-SAME: -> tensor<8x16x32xf32
  %1 = "ttir.logical_xor"(%arg0, %arg1, %0) : (tensor<8x16x32xf32>, tensor<1x16x32xf32>, tensor<8x16x32xf32>) -> tensor<8x16x32xf32>
  return %1 : tensor<8x16x32xf32>
}
