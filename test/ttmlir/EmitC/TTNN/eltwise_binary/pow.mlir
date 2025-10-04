// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @pow_tensor(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = ttir.empty() : tensor<64x128xf32>
  %1 = "ttir.pow"(%arg0, %arg1, %0) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %1 : tensor<64x128xf32>
}

func.func @pow_scalar_float(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = "ttir.constant"() <{value = dense<2.0> : tensor<64x128xf32>}> : () -> tensor<64x128xf32>
  %1 = ttir.empty() : tensor<64x128xf32>
  %2 = "ttir.pow"(%arg0, %0, %1) : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %2 : tensor<64x128xf32>
}

func.func @pow_scalar_integer(%arg0: tensor<64x128xf32>) -> tensor<64x128xf32> {
  %0 = "ttir.constant"() <{value = dense<3> : tensor<64x128xi32>}> : () -> tensor<64x128xi32>
  %1 = ttir.empty() : tensor<64x128xf32>
  %2 = "ttir.pow"(%arg0, %0, %1) : (tensor<64x128xf32>, tensor<64x128xi32>, tensor<64x128xf32>) -> tensor<64x128xf32>
  return %2 : tensor<64x128xf32>
}
