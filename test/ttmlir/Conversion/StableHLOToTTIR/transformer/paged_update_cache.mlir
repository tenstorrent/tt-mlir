
// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @paged_update_cache(%cache: tensor<128x3x32x256xbf16>, %input: tensor<1x8x12x256xbf16>, %update_index: tensor<8xi32>, %page_table: tensor<8x16xi32>) -> tensor<128x3x32x256xbf16> {
    // CHECK: "ttnn.paged_update_cache"
    %0 = stablehlo.custom_call @tt.paged_update_cache(%cache, %input, %update_index, %page_table) : (tensor<128x3x32x256xbf16>, tensor<1x8x12x256xbf16>, tensor<8xi32>, tensor<8x16xi32>) -> tensor<128x3x32x256xbf16>
    return %0 : tensor<128x3x32x256xbf16>
  }
}
