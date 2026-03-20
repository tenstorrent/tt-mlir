// RUN: ttmlir-opt --ttcore-register-device --ttnn-layout --ttnn-workaround -o %t %s
// RUN: FileCheck %s --input-file=%t

// Unit tests for ConcatOpPadDimRewritePattern.
// The pattern pads the last (unaligned) concat input to tile boundary, then
// slices the result back to the original shape. It fires when a partial-tile
// column would produce a NOC DMA write < 32 bytes.

module {
  // Test 1: Workaround SHOULD apply - last input unaligned with small NOC write
  // bf16 element = 2 bytes.  last-dim 8 => partial 8 elements => 16 bytes < 32.
  // Two inputs: first is tile-aligned (size 32), last is unaligned (size 8).
  func.func @concat_pad_dim_should_apply(
      %arg0: tensor<32x32xbf16>,
      %arg1: tensor<32x8xbf16>
  ) -> tensor<32x40xbf16> {
    // CHECK-LABEL: func.func @concat_pad_dim_should_apply
    // CHECK: "ttnn.pad"
    // CHECK: "ttnn.concat"
    // CHECK: "ttnn.slice_static"
    %0 = "ttnn.concat"(%arg0, %arg1) <{dim = 1 : si32}> : (tensor<32x32xbf16>, tensor<32x8xbf16>) -> tensor<32x40xbf16>
    return %0 : tensor<32x40xbf16>
  }

  // Test 2: Workaround should NOT apply - all inputs are tile-aligned
  func.func @concat_pad_dim_all_aligned(
      %arg0: tensor<32x32xbf16>,
      %arg1: tensor<32x32xbf16>
  ) -> tensor<32x64xbf16> {
    // CHECK-LABEL: func.func @concat_pad_dim_all_aligned
    // CHECK: "ttnn.concat"
    // CHECK-NOT: "ttnn.pad"
    // CHECK-NOT: "ttnn.slice_static"
    %0 = "ttnn.concat"(%arg0, %arg1) <{dim = 1 : si32}> : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x64xbf16>
    return %0 : tensor<32x64xbf16>
  }

  // Test 3: Workaround should NOT apply - non-final input is unaligned
  // (both inputs unaligned; pattern refuses to rewrite)
  func.func @concat_pad_dim_non_final_unaligned(
      %arg0: tensor<32x8xbf16>,
      %arg1: tensor<32x8xbf16>
  ) -> tensor<32x16xbf16> {
    // CHECK-LABEL: func.func @concat_pad_dim_non_final_unaligned
    // CHECK: "ttnn.concat"
    // CHECK-NOT: "ttnn.pad"
    // CHECK-NOT: "ttnn.slice_static"
    %0 = "ttnn.concat"(%arg0, %arg1) <{dim = 1 : si32}> : (tensor<32x8xbf16>, tensor<32x8xbf16>) -> tensor<32x16xbf16>
    return %0 : tensor<32x16xbf16>
  }

  // Test 4: Workaround should NOT apply - partial width is large enough
  // (f32 element = 4 bytes, last-dim 16 => partial 16 elems => 64 bytes >= 32)
  func.func @concat_pad_dim_large_partial_width(
      %arg0: tensor<32x32xf32>,
      %arg1: tensor<32x16xf32>
  ) -> tensor<32x48xf32> {
    // CHECK-LABEL: func.func @concat_pad_dim_large_partial_width
    // CHECK: "ttnn.concat"
    // CHECK-NOT: "ttnn.pad"
    // CHECK-NOT: "ttnn.slice_static"
    %0 = "ttnn.concat"(%arg0, %arg1) <{dim = 1 : si32}> : (tensor<32x32xf32>, tensor<32x16xf32>) -> tensor<32x48xf32>
    return %0 : tensor<32x48xf32>
  }
}
