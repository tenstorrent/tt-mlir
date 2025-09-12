// This test checks that the (TTIR to EmitPy pipeline) is equivalent to (TTIR to TTNN BE pipeline + TTNN BE to EmitPy pipeline).
// The `diff` command will return 0 if files are identical, otherwise it will return the diff, which will make `llvm-lit` treat the test as failed.
//
// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path%" -o %t_direct.mlir %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --ttnn-backend-to-emitpy-pipeline -o %t_indirect.mlir %s
// RUN: diff %t_direct.mlir %t_indirect.mlir
// RUN: FileCheck %s --input-file=%t_direct.mlir

// CHECK: func.func @add(%arg0: !emitpy.opaque<"[ttnn.Tensor]">) -> !emitpy.opaque<"[ttnn.Tensor]">
// CHECK: func.func @create_inputs_for_add() -> !emitpy.opaque<"[ttnn.Tensor]">
// CHECK: func.func @main() -> i32
func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
