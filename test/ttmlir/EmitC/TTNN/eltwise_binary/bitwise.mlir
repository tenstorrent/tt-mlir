// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

  func.func @bitwise_and(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = ttir.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_and"(%arg0, %arg1, %0) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    return %1 : tensor<64x128xi32>
  }

  func.func @bitwise_or(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = ttir.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_or"(%arg0, %arg1, %0) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    return %1 : tensor<64x128xi32>
  }

  func.func @bitwise_xor(%arg0: tensor<64x128xi32>, %arg1: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = ttir.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_xor"(%arg0, %arg1, %0) : (tensor<64x128xi32>, tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    return %1 : tensor<64x128xi32>
  }
