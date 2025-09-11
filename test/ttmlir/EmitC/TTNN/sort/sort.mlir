// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64x128xi16>) {
  %0 = ttir.empty() : tensor<64x128xbf16>
  %1 = ttir.empty() : tensor<64x128xi16>
  %2, %3 = "ttir.sort"(%arg0, %0, %1) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xi16>) -> (tensor<64x128xbf16>, tensor<64x128xi16>)
  return %2, %3 : tensor<64x128xbf16>, tensor<64x128xi16>
}
