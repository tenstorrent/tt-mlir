// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @logicalrightshift(%arg0: tensor<5xui32>, %arg1: tensor<5xui32>) -> tensor<5xui32> {
  %0 = ttir.empty() : tensor<5xui32>
  %1 = "ttir.logical_right_shift"(%arg0, %arg1, %0) : (tensor<5xui32>, tensor<5xui32>, tensor<5xui32>) -> tensor<5xui32>
  return %1 : tensor<5xui32>
}
