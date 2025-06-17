// RUN: ttmlir-opt --ttir-to-emitpy-pipeline %s | ttmlir-translate --mlir-to-python > %t.mlir

func.func @matmul(%arg0: tensor<64x128xbf16>, %arg1: tensor<128x96xbf16>) -> tensor<64x96xbf16> {
  %0 = ttir.empty() : tensor<64x96xbf16>
  %1 = "ttir.matmul"(%arg0, %arg1, %0) : (tensor<64x128xbf16>, tensor<128x96xbf16>, tensor<64x96xbf16>) -> tensor<64x96xbf16>
  return %1 : tensor<64x96xbf16>
}
