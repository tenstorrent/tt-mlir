// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %basename_t.ttnn
// RUN: ttmlir-opt --ttnn-tuplify-tensors --convert-ttnn-to-emitc %t.mlir > %t2.mlir
// RUN: ttmlir-translate --mlir-to-cpp %t2.mlir > %basename_t.cpp
// UNSUPPORTED: true
// This test is failing due to not handling multiple outputs in TTNN to EmitC
// conversion pipeline.
// tt-mlir issue: https://github.com/tenstorrent/tt-mlir/issues/4045

func.func @test_sort(%arg0: tensor<64x128xbf16>) -> (tensor<64x128xbf16>, tensor<64x128xi16>) {
  %0 = ttir.empty() : tensor<64x128xbf16>
  %1 = ttir.empty() : tensor<64x128xi16>
  %2, %3 = "ttir.sort"(%arg0, %0, %1) : (tensor<64x128xbf16>, tensor<64x128xbf16>, tensor<64x128xi16>) -> (tensor<64x128xbf16>, tensor<64x128xi16>)
  return %2, %3 : tensor<64x128xbf16>, tensor<64x128xi16>
}
