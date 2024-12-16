// RUN: ttmlir-opt --ttir-to-emitc-pipeline="system-desc-path=%system_desc_path%" %s > %direct.mlir
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --convert-ttnn-to-emitc %s > %indirect.mlir
// RUN: diff %direct.mlir %indirect.mlir
//
// This test checks that the (TTIR to EmitC pipeline) is equivalent to (TTIR to TTNN pipeline + dialect conversion from TTNN to EmitC).
// The `diff` command will return 0 if files are identical, otherwise it will return the diff, which will make `llvm-lit` treat the test as failed.

func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = tensor.empty() : tensor<64x128xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) <{operandSegmentSizes = array<i32: 2, 1>}> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
