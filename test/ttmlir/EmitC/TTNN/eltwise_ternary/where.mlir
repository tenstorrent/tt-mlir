// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @test_where(%arg0: tensor<13x37xbf16>, %arg1: tensor<13x37xbf16>) -> tensor<13x37xbf16> {
  %0 = ttir.empty() : tensor<13x37xbf16>
  %1 = "ttir.eq"(%arg0, %arg1, %0) : (tensor<13x37xbf16>, tensor<13x37xbf16>, tensor<13x37xbf16>) -> tensor<13x37xbf16>
  %2 = ttir.empty() : tensor<13x37xbf16>
  %3 = "ttir.where"(%1, %arg0, %arg1, %2) : (tensor<13x37xbf16>, tensor<13x37xbf16>, tensor<13x37xbf16>, tensor<13x37xbf16>) -> tensor<13x37xbf16>
  return %3 : tensor<13x37xbf16>
}
