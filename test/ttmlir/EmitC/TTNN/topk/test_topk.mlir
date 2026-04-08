// RUN: ttmlir-opt --ttir-to-ttnn-common-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
//
// RUN: ttmlir-opt --ttnn-common-to-runtime-pipeline -o %t_rt.mlir %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t_rt.mlir
//
// RUN: ttmlir-opt --ttnn-common-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @test_basic_top_k(%input: tensor<2x3x32x128xbf16>) -> (tensor<2x3x32x5xbf16>, tensor<2x3x32x5xi32>) {
  %values, %indices = "ttir.topk"(%input) { k = 5 : i32} : (tensor<2x3x32x128xbf16>) -> (tensor<2x3x32x5xbf16>, tensor<2x3x32x5xi32>)
  return %values, %indices : tensor<2x3x32x5xbf16>, tensor<2x3x32x5xi32>
}
