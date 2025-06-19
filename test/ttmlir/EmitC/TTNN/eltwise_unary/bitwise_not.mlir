// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline="target-dylib=true" %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @bitwise_not(%arg0: tensor<64x128xi32>) -> tensor<64x128xi32> {
    %0 = ttir.empty() : tensor<64x128xi32>
    %1 = "ttir.bitwise_not"(%arg0, %0) : (tensor<64x128xi32>, tensor<64x128xi32>) -> tensor<64x128xi32>
    return %1 : tensor<64x128xi32>
}
