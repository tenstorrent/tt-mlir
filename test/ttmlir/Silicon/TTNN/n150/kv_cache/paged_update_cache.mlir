// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %t.ttnn %t.mlir
module {
  func.func @paged_update_cache(%cache: tensor<128x3x32x256xbf16>, %input: tensor<1x8x12x256xbf16>, %update_index: tensor<8xi32>, %page_table: tensor<8x16xi32>) -> tensor<128x3x32x256xbf16> {
    // CHECK: "ttnn.paged_update_cache"
    %0 = "ttir.paged_update_cache"(%cache, %input, %update_index, %page_table) : (tensor<128x3x32x256xbf16>, tensor<1x8x12x256xbf16>, tensor<8xi32>, tensor<8x16xi32>) -> tensor<128x3x32x256xbf16>
    return %0 : tensor<128x3x32x256xbf16>
  }
}
