// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // Exercise the optional mask and all forward/backward results through the
  // optimizer and OpModel validation.
  func.func @sdpa_forward_backward(
      %grad_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>,
      %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>,
      %mask: tensor<1x1x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
          tensor<1x8x64x64xbf16>) {
    // CHECK-LABEL: func.func @sdpa_forward_backward
    // CHECK: %[[OUTPUT:.*]], %[[INTERMEDIATES:.*]] = "ttnn.sdpa_fw"
    // CHECK: %[[GQ:.*]], %[[GK:.*]], %[[GV:.*]] = "ttnn.sdpa_bw"(%{{.*}}, %[[OUTPUT]], %{{.*}}, %{{.*}}, %{{.*}}, %[[INTERMEDIATES]],
    // CHECK: return %[[GQ]], %[[GK]], %[[GV]]
    %output, %intermediates = "ttir.sdpa_fw"(%query, %key, %value, %mask) <{
        mask_type = #ttcore.attention_mask_type<arbitrary>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = true}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
    %grad_query, %grad_key, %grad_value = "ttir.sdpa_bw"(
        %grad_output, %output, %query, %key, %value, %intermediates, %mask) <{
        mask_type = #ttcore.attention_mask_type<arbitrary>,
        dropout_probability = 0.000000e+00 : f32}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>,
           tensor<1x1x64x64xbf16>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
              tensor<1x8x64x64xbf16>)
    return %grad_query, %grad_key, %grad_value
        : tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
          tensor<1x8x64x64xbf16>
  }

  // SDPA forward requires interleaved inputs. Verify that a compute-produced
  // query is resharded before reaching the op.
  func.func @sdpa_fw_query_from_l1_producer(
      %q0: tensor<1x8x128x64xbf16>,
      %q1: tensor<1x8x128x64xbf16>,
      %key: tensor<1x8x128x64xbf16>,
      %value: tensor<1x8x128x64xbf16>)
      -> tensor<1x8x128x64xbf16> {
    // CHECK-LABEL: func.func @sdpa_fw_query_from_l1_producer
    // CHECK: %[[MUL:.*]] = "ttnn.multiply"
    // CHECK: %[[INTERLEAVED:.*]] = "ttnn.to_memory_config"(%[[MUL]])
    // CHECK: %[[RESHARD:.*]] = "ttnn.to_memory_config"(%[[INTERLEAVED]])
    // CHECK: "ttnn.sdpa_fw"(%[[RESHARD]],
    %query = "ttir.multiply"(%q0, %q1)
        : (tensor<1x8x128x64xbf16>, tensor<1x8x128x64xbf16>)
          -> tensor<1x8x128x64xbf16>
    %output = "ttir.sdpa_fw"(%query, %key, %value) <{
        mask_type = #ttcore.attention_mask_type<causal>,
        dropout_probability = 0.000000e+00 : f32,
        return_intermediates = false}>
        : (tensor<1x8x128x64xbf16>, tensor<1x8x128x64xbf16>,
           tensor<1x8x128x64xbf16>)
          -> tensor<1x8x128x64xbf16>
    return %output : tensor<1x8x128x64xbf16>
  }
}
