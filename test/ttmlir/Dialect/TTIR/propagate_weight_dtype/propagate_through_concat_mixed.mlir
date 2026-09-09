// RUN: ttmlir-opt --ttir-propagate-weight-dtype --verify-diagnostics %s | FileCheck %s

// Test that when concatenated weights carry differing overrides, the fused
// matmul gets the highest-precision (most bits) dtype so no constituent loses
// precision, and a warning is emitted. Here bfp_bf4 (4 bits) and bfp_bf8 (8
// bits) merge to bfp_bf8.

module {
  // CHECK-LABEL: func.func @propagate_through_concat_mixed
  func.func @propagate_through_concat_mixed(
    %arg0: tensor<32x512xbf16>,
    %q: tensor<512x256xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.weight_dtype = "bfp_bf4"},
    %k: tensor<512x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.weight_dtype = "bfp_bf8"},
    %v: tensor<512x128xbf16> {ttcore.argument_type = #ttcore.argument_type<parameter>, ttcore.weight_dtype = "bfp_bf4"}
  ) -> tensor<32x512xbf16> {
    %0 = "ttir.concat"(%q, %k, %v) <{dim = 1 : si32}> : (tensor<512x256xbf16>, tensor<512x128xbf16>, tensor<512x128xbf16>) -> tensor<512x512xbf16>
    // CHECK: "ttir.matmul"
    // CHECK-SAME: ttcore.weight_dtype = "bfp_bf8"
    // expected-warning @+1 {{fusing matmuls with differing weight dtype overrides}}
    %1 = "ttir.matmul"(%arg0, %0) : (tensor<32x512xbf16>, tensor<512x512xbf16>) -> tensor<32x512xbf16>
    return %1 : tensor<32x512xbf16>
  }
}
