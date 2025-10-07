// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-backend-to-emitc-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

func.func @update_cache(%arg0: tensor<1x8x16x128xbf16>, %arg1: tensor<1x8x1x128xbf16>, %arg2: tensor<1xui32>) -> tensor<1x8x16x128xbf16> {
  %0 = "ttir.update_cache"(%arg0, %arg1, %arg2) <{batch_offset = 0 : i32}> : (tensor<1x8x16x128xbf16>, tensor<1x8x1x128xbf16>, tensor<1xui32>) -> tensor<1x8x16x128xbf16>
  return %0 : tensor<1x8x16x128xbf16>
}
