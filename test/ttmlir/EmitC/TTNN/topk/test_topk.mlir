// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @test_basic_top_k(%input: tensor<2x3x32x128xf32>) -> (tensor<2x3x32x5xf32>, tensor<2x3x32x5xi32>) {
  %values, %indices = "ttir.topk"(%input) { k = 5 : i32} : (tensor<2x3x32x128xf32>) -> (tensor<2x3x32x5xf32>, tensor<2x3x32x5xi32>)
  return %values, %indices : tensor<2x3x32x5xf32>, tensor<2x3x32x5xi32>
}
