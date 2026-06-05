// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Batched fill: fill_value dim0 = num_users (4), batch_indices = [num_users].
  func.func @paged_fill_cache_batched(%cache: tensor<128x12x32x256xbf16>, %fill_value: tensor<4x12x65x256xbf16>, %page_table: tensor<8x16xi32>, %batch_indices: tensor<4xi64>) -> tensor<128x12x32x256xbf16> {
    // CHECK: "ttir.paged_fill_cache"
    // CHECK-SAME: tensor<4x12x65x256xbf16>
    %0 = stablehlo.custom_call @tt.paged_fill_cache(%cache, %fill_value, %page_table, %batch_indices) : (tensor<128x12x32x256xbf16>, tensor<4x12x65x256xbf16>, tensor<8x16xi32>, tensor<4xi64>) -> tensor<128x12x32x256xbf16>
    return %0 : tensor<128x12x32x256xbf16>
  }
}
