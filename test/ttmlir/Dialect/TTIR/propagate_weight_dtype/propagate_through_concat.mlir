// RUN: ttmlir-opt --ttir-propagate-weight-dtype %s | FileCheck %s

// Test that weight_dtype propagates through a concat (e.g. a fused QKV weight
// built by SharedLHSMatmulFusion) when all concatenated weights agree.

module {
  // CHECK-LABEL: func.func @propagate_through_concat
  func.func @propagate_through_concat(
    %arg0: tensor<32x512xbf16>,
    %q: tensor<512x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.weight_dtype = "bfp_bf4"},
    %k: tensor<512x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.weight_dtype = "bfp_bf4"},
    %v: tensor<512x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.weight_dtype = "bfp_bf4"}
  ) -> tensor<32x512xbf16> {
    %0 = "ttir.concat"(%q, %k, %v) <{dim = 1 : si32}> : (tensor<512x256xbf16>, tensor<512x128xbf16>, tensor<512x128xbf16>) -> tensor<512x512xbf16>
    // CHECK: "ttir.matmul"
    // CHECK-SAME: ttcore.weight_dtype = "bfp_bf4"
    %1 = "ttir.matmul"(%arg0, %0) : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
    return %1 : tensor<32x512xbf16>
  }
}
