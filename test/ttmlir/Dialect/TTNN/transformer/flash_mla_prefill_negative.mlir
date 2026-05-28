// RUN: not ttmlir-opt --split-input-file %s 2>&1 | FileCheck %s

// Negative tests for the ttnn.flash_mla_prefill verifier covering checks not
// already enforced by the TTIR verifier.

// Verify that query and key must have the same element type.
module {
  func.func @flash_mla_prefill_query_key_dtype(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xf32>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttnn.flash_mla_prefill' op Query and key must have the same element type
    %0 = "ttnn.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xf32>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that query and value must have the same element type.
module {
  func.func @flash_mla_prefill_query_value_dtype(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>, %value: tensor<1x1x32x64xf32>) -> tensor<1x16x32x64xbf16> {
    // CHECK: error: 'ttnn.flash_mla_prefill' op Query and value must have the same element type
    %0 = "ttnn.flash_mla_prefill"(%query, %key, %value) <{operandSegmentSizes = array<i32: 1, 1, 1, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>, tensor<1x1x32x64xf32>) -> tensor<1x16x32x64xbf16>
    return %0 : tensor<1x16x32x64xbf16>
  }
}

// -----

// Verify that query and result must have the same element type.
module {
  func.func @flash_mla_prefill_query_result_dtype(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xf32> {
    // CHECK: error: 'ttnn.flash_mla_prefill' op Query and result must have the same element type
    %0 = "ttnn.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 64 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x64xf32>
    return %0 : tensor<1x16x32x64xf32>
  }
}

// -----

// Verify that head_dim_v must be greater than 0.
module {
  func.func @flash_mla_prefill_head_dim_zero(%query: tensor<1x16x32x128xbf16>, %key: tensor<1x1x32x128xbf16>) -> tensor<1x16x32x0xbf16> {
    // CHECK: error: 'ttnn.flash_mla_prefill' op head_dim_v must be greater than 0
    %0 = "ttnn.flash_mla_prefill"(%query, %key) <{operandSegmentSizes = array<i32: 1, 1, 0, 0>, head_dim_v = 0 : ui32, is_causal = true}> : (tensor<1x16x32x128xbf16>, tensor<1x1x32x128xbf16>) -> tensor<1x16x32x0xbf16>
    return %0 : tensor<1x16x32x0xbf16>
  }
}
