// RUN: ttmlir-opt --ttir-to-emitc-pipeline="system-desc-path=%system_desc_path%" %s > %t_direct.mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --tt-unwrap-device-module --ttnn-create-input-gens --convert-ttnn-to-emitc %s > %t_indirect.mlir
// RUN: diff %t_direct.mlir %t_indirect.mlir
// RUN: FileCheck %s --input-file=%t_direct.mlir
//
// This test checks that the (TTIR to EmitC pipeline) is equivalent to (TTIR to TTNN pipeline + dialect conversion from TTNN to EmitC).
// The `diff` command will return 0 if files are identical, otherwise it will return the diff, which will make `llvm-lit` treat the test as failed.

// CHECK: func.func @add(%arg0: !emitc.opaque<"::ttnn::Tensor">, %arg1: !emitc.opaque<"::ttnn::Tensor">) -> !emitc.opaque<"::ttnn::Tensor"
// CHECK: func.func @createInputsFor_add() -> (!emitc.opaque<"::ttnn::Tensor">, !emitc.opaque<"::ttnn::Tensor">)
// CHECK: func.func @main() -> i32
func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
