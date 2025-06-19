// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline="target-dylib=true" %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp

func.func @test_where(%arg0: tensor<13x37xbf16>, %arg1: tensor<13x37xbf16>) -> tensor<13x37xbf16> {
  %0 = ttir.empty() : tensor<13x37xbf16>
  %1 = "ttir.eq"(%arg0, %arg1, %0) : (tensor<13x37xbf16>, tensor<13x37xbf16>, tensor<13x37xbf16>) -> tensor<13x37xbf16>
  %2 = ttir.empty() : tensor<13x37xbf16>
  %3 = "ttir.where"(%1, %arg0, %arg1, %2) : (tensor<13x37xbf16>, tensor<13x37xbf16>, tensor<13x37xbf16>, tensor<13x37xbf16>) -> tensor<13x37xbf16>
  return %3 : tensor<13x37xbf16>
}
