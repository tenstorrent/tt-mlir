
// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @paged_fill_cache(%cache: tensor<128x4x32x256xbf16>, %fill_value: tensor<1x12x65x256xbf16>, %page_table: tensor<8x16xi32>, %batch_offset: tensor<1xi64>) -> tensor<128x4x32x256xbf16> {
    // CHECK: "ttir.paged_fill_cache"
    %0 = stablehlo.custom_call @tt.paged_fill_cache(%cache, %fill_value, %page_table, %batch_offset) : (tensor<128x4x32x256xbf16>, tensor<1x12x65x256xbf16>, tensor<8x16xi32>, tensor<1xi64>) -> tensor<128x4x32x256xbf16>
    return %0 : tensor<128x4x32x256xbf16>
  }
}
