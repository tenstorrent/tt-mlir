// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="mesh-shape=1,4" -o %t %s
// RUN: FileCheck %s --input-file=%t
// Unit tests for ttnn moe_gpt op

// Verify lowering of ttir moe_gpt to ttnn moe_gpt

module attributes {} {
  // CHECK-LABEL: moe_gpt_basic
  func.func @moe_gpt_basic(
      %input: tensor<128x2880xbf16>,
      %indices: tensor<128x4xui16>,
      %scores: tensor<128x4xbf16>,
      %mapping: tensor<1x16xui16>,
      %w0w1: tensor<12x1x4x4x2880x128xbf16>,
      %w2: tensor<12x1x4x2x2880x128xbf16>
  ) -> (tensor<1x16xui32>, tensor<1x512xui32>, tensor<4x129xui32>, tensor<12x2x32x2880xbf16>, tensor<12x2x32x2880xbf16>) {
    // CHECK: "ttnn.moe_gpt"
    %0, %1, %2, %3, %4 = "ttir.moe_gpt"(%input, %indices, %scores, %mapping, %w0w1, %w2)
        <{output_height_shard_dim = 4 : ui32,
          output_width_shard_dim = 3 : ui32,
          hidden_size = 2880 : ui32,
          cluster_axis = 0 : ui32}>
        : (tensor<128x2880xbf16>, tensor<128x4xui16>, tensor<128x4xbf16>,
           tensor<1x16xui16>, tensor<12x1x4x4x2880x128xbf16>,
           tensor<12x1x4x2x2880x128xbf16>)
        -> (tensor<1x16xui32>, tensor<1x512xui32>, tensor<4x129xui32>,
            tensor<12x2x32x2880xbf16>, tensor<12x2x32x2880xbf16>)
    return %0, %1, %2, %3, %4
        : tensor<1x16xui32>, tensor<1x512xui32>, tensor<4x129xui32>,
          tensor<12x2x32x2880xbf16>, tensor<12x2x32x2880xbf16>
  }
}
