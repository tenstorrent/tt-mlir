// UNSUPPORTED: true
// RUN: ttmlir-opt --ttir-to-ttmetal-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer %t.mlir > %t.ttm

func.func public @add5(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: %[[C:.*]] = "ttmetal.create_buffer"[[C:.*]]
  // CHECK: %[[C:.*]] = "ttmetal.enqueue_write_buffer"[[C:.*]]
  %0 = "ttir.constant"() <{value = dense<5.0> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
  %1 = tensor.empty() : tensor<32x32xf32>
  %2 = "ttir.add"(%arg0, %0, %1) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}
