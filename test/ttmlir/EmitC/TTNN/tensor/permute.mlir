// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline="target-dylib=true" %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @permute(%arg0: tensor<1x4x32x64xf32>) -> tensor<4x32x64x1xf32> {
  %0 = ttir.empty() : tensor<4x32x64x1xf32>
  %1 = "ttir.permute"(%arg0, %0) <{permutation = array<i64: 1, 2, 3, 0>}> : (tensor<1x4x32x64xf32>, tensor<4x32x64x1xf32>) -> tensor<4x32x64x1xf32>
  return %1 : tensor<4x32x64x1xf32>
}
