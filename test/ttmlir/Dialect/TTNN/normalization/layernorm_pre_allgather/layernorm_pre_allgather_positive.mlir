// RUN: ttmlir-opt --split-input-file %s | FileCheck %s
// Unit tests for ttnn layernorm_pre_allgather op

// -----

// Verify basic layernorm_pre_allgather round-trips correctly.
module attributes {} {
  // CHECK-LABEL: layernorm_pre_allgather_basic
  func.func @layernorm_pre_allgather_basic(%arg0: tensor<1x1x32x1024xbf16>, %arg1: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16> {
    // CHECK: "ttnn.layernorm_pre_allgather"
    %0 = "ttnn.layernorm_pre_allgather"(%arg0, %arg1) : (tensor<1x1x32x1024xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
}

// -----

// Verify layernorm_pre_allgather with optional dtype attribute.
module attributes {} {
  // CHECK-LABEL: layernorm_pre_allgather_with_dtype
  func.func @layernorm_pre_allgather_with_dtype(%arg0: tensor<1x1x32x1024xbf16>, %arg1: tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16> {
    // CHECK: "ttnn.layernorm_pre_allgather"
    // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
    %0 = "ttnn.layernorm_pre_allgather"(%arg0, %arg1) {dtype = #tt.supportedDataTypes<bf16>} : (tensor<1x1x32x1024xbf16>, tensor<1x1x32x32xbf16>) -> tensor<1x1x32x64xbf16>
    return %0 : tensor<1x1x32x64xbf16>
  }
}
