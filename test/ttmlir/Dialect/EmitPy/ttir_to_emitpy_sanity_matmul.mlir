// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="system-desc-path=%system_desc_path%" %s > %t_direct.mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --ttcore-unwrap-device-module --ttnn-tuplify-tensors --ttnn-create-input-gens --convert-ttnn-to-emitpy %s > %t_indirect.mlir
// RUN: diff %t_direct.mlir %t_indirect.mlir
// RUN: FileCheck %s --input-file=%t_direct.mlir
//
// This test checks that the (TTIR to EmitPy pipeline) is equivalent to (TTIR to TTNN pipeline + dialect conversion from TTNN to EmitPy).
// The `diff` command will return 0 if files are identical, otherwise it will return the diff, which will make `llvm-lit` treat the test as failed.

// CHECK: func.func @matmul(%arg0: !emitpy.opaque<"vector<ttnn.Tensor>">) -> !emitpy.opaque<"vector<ttnn.Tensor>">
// CHECK: func.func @create_inputs_for_matmul() -> !emitpy.opaque<"vector<ttnn.Tensor>">
// CHECK: func.func @main() -> i32
func.func @matmul(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
  %0 = ttir.empty() : tensor<64x96xbf16>
  %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
  return %1 : tensor<64x96xbf16>
}
