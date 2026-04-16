// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path% mesh-shape=1,4" -o %t.mlir %s
// RUN: ttmlir-translate --ttnn-to-flatbuffer -o %basename_t.ttnn %t.mlir
// RUN: ttmlir-opt --ttnn-to-emitc-device-pipeline -o %t2.mlir %t.mlir
// RUN: ttmlir-translate --mlir-to-cpp -o %basename_t.cpp %t2.mlir

module attributes {} {
  func.func @moe_gpt(
      %input: tensor<128x2880xbf16>,
      %indices: tensor<128x4xui16>,
      %scores: tensor<128x4xbf16>,
      %mapping: tensor<1x16xui16>,
      %w0w1: tensor<12x1x4x4x2880x128xbf16>,
      %w2: tensor<12x1x4x2x2880x128xbf16>
  ) -> (tensor<1x16xui32>, tensor<1x512xui32>, tensor<4x129xui32>, tensor<12x2x32x2880xbf16>, tensor<12x2x32x2880xbf16>) {
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
