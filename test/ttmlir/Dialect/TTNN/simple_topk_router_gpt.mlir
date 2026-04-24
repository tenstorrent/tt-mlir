// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module attributes {} {
  // Test fused matmul + top-k router with k=4, num_experts=128.
  // Output shapes use the semantic [B, k] dimensions.
  // The k_padded hardware constraint is handled by a TTNN workaround pass.
  func.func @test_topk_router_gpt(
      %input:  tensor<32x64xbf16>,
      %weight: tensor<64x128xbf16>,
      %bias:   tensor<32x128xbf16>
  ) -> (tensor<32x4xui16>, tensor<32x4xbf16>) {
    // CHECK-LABEL: func.func @test_topk_router_gpt
    // CHECK: "ttnn.topk_router_gpt"
    // CHECK-SAME: <{k = 8 : i32, num_experts = 128 : i32}>
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: ends = [32 : i32, 4 : i32]
    // CHECK: "ttnn.slice_static"
    // CHECK-SAME: ends = [32 : i32, 4 : i32]
    %indices, %weights = "ttir.topk_router_gpt"(%input, %weight, %bias)
        <{k = 4 : i32, num_experts = 128 : i32}>
        : (tensor<32x64xbf16>, tensor<64x128xbf16>, tensor<32x128xbf16>)
          -> (tensor<32x4xui16>, tensor<32x4xbf16>)
    return %indices, %weights : tensor<32x4xui16>, tensor<32x4xbf16>
  }
}
