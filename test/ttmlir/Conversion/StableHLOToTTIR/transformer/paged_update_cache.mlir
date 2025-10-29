
// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  func.func @paged_update_cache(%cache: tensor<128x1x32x128xbf16>, %fill_value: tensor<1x8x32x128xbf16>, %update_index: tensor<8xi64>, %page_table: tensor<8x16xi32>) -> tensor<128x1x32x128xbf16> {
    // CHECK: "ttir.paged_update_cache"
    %0 = stablehlo.custom_call @tt.paged_update_cache(%cache, %fill_value, %update_index, %page_table) : (tensor<128x1x32x128xbf16>, tensor<1x8x32x128xbf16>, tensor<8xi64>, tensor<8x16xi32>) -> tensor<128x1x32x128xbf16>
    return %0 : tensor<128x1x32x128xbf16>
  }
}
