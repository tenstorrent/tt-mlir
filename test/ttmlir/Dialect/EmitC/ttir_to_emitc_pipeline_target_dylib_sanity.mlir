// This test checks that the (TTIR to EmitC pipeline with target dylib) is equivalent to (TTIR to TTNN BE pipeline + TTNN BE to EmitC pipeline with target dylib).
// The `diff` command will return 0 if files are identical, otherwise it will return the diff, which will make `llvm-lit` treat the test as failed.
//
// RUN: ttmlir-opt --ttir-to-emitc-pipeline="target-dylib=true system-desc-path=%system_desc_path%" -o %t_direct.mlir %s
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" --ttcore-unwrap-device-module --ttnn-backend-to-emitc-pipeline="target-dylib=true" -o %t_indirect.mlir %s
// RUN: diff %t_direct.mlir %t_indirect.mlir
// RUN: FileCheck %s --input-file=%t_direct.mlir
//

// CHECK: func.func @add(%arg0: !emitc.opaque<"::std::vector<::ttnn::Tensor>">) -> !emitc.opaque<"::std::vector<::ttnn::Tensor>">
func.func @add(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.add"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  // CHECK: return %{{[0-9]+}} : !emitc.opaque<"::std::vector<::ttnn::Tensor>">
  return %1 : tensor<64x128xf32>
}
