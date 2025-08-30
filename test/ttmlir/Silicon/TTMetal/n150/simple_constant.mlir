// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir
// RUN: ttrt run %t.ttm

func.func public @add5(%arg0: tensor<32x32xf32>) -> tensor<32x32xf32> {
  // CHECK: = "ttmetal.create_buffer"
  // CHECK: "ttmetal.enqueue_write_buffer"
  %0 = "ttir.constant"() <{value = dense<5.0> : tensor<32x32xf32>}> : () -> tensor<32x32xf32>
  %1 = ttir.empty() : tensor<32x32xf32>
  %2 = "ttir.add"(%arg0, %0, %1) : (tensor<32x32xf32>, tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %2 : tensor<32x32xf32>
}
