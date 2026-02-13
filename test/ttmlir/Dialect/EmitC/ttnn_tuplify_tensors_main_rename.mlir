// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path%" --ttcore-unwrap-device-module --ttnn-tuplify-tensors -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// If the function is named `main`, it has to be renamed, otherwise compiler
// will complain about the parameters and return type.
// In `TTNNTuplifyTensors` it is renamed to `_main`.

// CHECK-NOT: func.func @main
// CHECK-LABEL: func.func @_main
// CHECK-NOT: func.func @main
func.func @main(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %1 = "ttir.add"(%arg0, %arg1) : (tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}
