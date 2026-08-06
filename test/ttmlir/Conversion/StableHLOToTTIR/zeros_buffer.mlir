// REQUIRES: stablehlo
// RUN: ttmlir-opt --stablehlo-to-ttir-pipeline -split-input-file -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Two calls with identical result types must both survive as separate ops.
  // has_side_effect = true is what keeps them apart upstream; the Allocate
  // effect on ttir.zeros_buffer is what keeps tt-mlir's own CSE from
  // merging them again.
  func.func @create_two_caches() -> (tensor<8x16x32x128xbf16>, tensor<8x16x32x128xbf16>) {
    // CHECK-LABEL: func.func @create_two_caches
    // CHECK: %[[K:.*]] = "ttir.zeros_buffer"()
    // CHECK-SAME: shape = array<i32: 8, 16, 32, 128>
    // CHECK: %[[V:.*]] = "ttir.zeros_buffer"()
    // CHECK-SAME: shape = array<i32: 8, 16, 32, 128>
    // CHECK: return %[[K]], %[[V]]
    %k = stablehlo.custom_call @tt.zeros_buffer() {api_version = 0 : i32, has_side_effect = true} : () -> tensor<8x16x32x128xbf16>
    %v = stablehlo.custom_call @tt.zeros_buffer() {api_version = 0 : i32, has_side_effect = true} : () -> tensor<8x16x32x128xbf16>
    return %k, %v : tensor<8x16x32x128xbf16>, tensor<8x16x32x128xbf16>
  }

  func.func @create_cache_f32() -> tensor<64x128xf32> {
    // CHECK-LABEL: func.func @create_cache_f32
    // CHECK: "ttir.zeros_buffer"()
    // CHECK-SAME: shape = array<i32: 64, 128>
    // CHECK-SAME: -> tensor<64x128xf32>
    %0 = stablehlo.custom_call @tt.zeros_buffer() {api_version = 0 : i32, has_side_effect = true} : () -> tensor<64x128xf32>
    return %0 : tensor<64x128xf32>
  }
}
