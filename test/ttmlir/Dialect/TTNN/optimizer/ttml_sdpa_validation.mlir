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
    %output, %intermediates = "ttcore.composite"(%query, %key, %value, %mask) <{
        composite_name = "sdpa_fw",
        decomposition = @sdpa_fw_bf16_decomposition,
        composite_attributes = {
          mask_type = #ttcore.attention_mask_type<arbitrary>,
          dropout_probability = 0.000000e+00 : f32,
          return_intermediates = true}}>
        : (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
           tensor<1x8x64x64xbf16>, tensor<1x1x64x64xbf16>)
          -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>)
    %grad_query, %grad_key, %grad_value = "ttcore.composite"(
        %grad_output, %output, %query, %key, %value, %intermediates, %mask) <{
        composite_name = "sdpa_bw",
        decomposition = @sdpa_bw_bf16_decomposition,
        composite_attributes = {
          mask_type = #ttcore.attention_mask_type<arbitrary>,
          dropout_probability = 0.000000e+00 : f32}}>
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

  // The backing kernel requires bf16 inputs and output. Verify that f32 values
  // are cast to bf16 for the kernel and that its output is cast back to f32.
  // The optional intermediates must remain f32.
  func.func @sdpa_forward_f32(
      %query: tensor<1x8x64x64xf32>,
      %key: tensor<1x8x64x64xf32>,
      %value: tensor<1x8x64x64xf32>,
      %mask: tensor<1x1x64x64xf32>)
      -> tensor<1x8x64x64xf32> {
    // CHECK-LABEL: func.func @sdpa_forward_f32
    // CHECK: %[[QUERY_BF16:.*]] = "ttnn.typecast"(%{{.*}}) : {{.*}} -> tensor<1x8x64x64xbf16
    // CHECK: %[[KEY_BF16:.*]] = "ttnn.typecast"(%{{.*}}) : {{.*}} -> tensor<1x8x64x64xbf16
    // CHECK: %[[VALUE_BF16:.*]] = "ttnn.typecast"(%{{.*}}) : {{.*}} -> tensor<1x8x64x64xbf16
    // CHECK: %[[MASK_BF16:.*]] = "ttnn.typecast"(%{{.*}}) : {{.*}} -> tensor<1x1x64x64xbf16
    // CHECK: %[[OUTPUT_BF16:.*]], %[[INTERMEDIATES_F32:.*]] = "ttnn.sdpa_fw"(%[[QUERY_BF16]], %[[KEY_BF16]], %[[VALUE_BF16]], %[[MASK_BF16]])
    // CHECK-SAME: -> (tensor<1x8x64x64xbf16
    // CHECK-SAME: tensor<1x8x64x32xf32
    // CHECK: %[[OUTPUT_F32:.*]] = "ttnn.typecast"(%[[OUTPUT_BF16]]) : {{.*}} -> tensor<1x8x64x64xf32
    // CHECK: return %[[OUTPUT_F32]]
    %output, %intermediates = "ttcore.composite"(%query, %key, %value, %mask) <{
        composite_name = "sdpa_fw",
        decomposition = @sdpa_fw_f32_decomposition,
        composite_attributes = {
          mask_type = #ttcore.attention_mask_type<arbitrary>,
          dropout_probability = 0.000000e+00 : f32,
          return_intermediates = true}}>
        : (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>,
           tensor<1x8x64x64xf32>, tensor<1x1x64x64xf32>)
          -> (tensor<1x8x64x64xf32>, tensor<1x8x64x32xf32>)
    return %output : tensor<1x8x64x64xf32>
  }

  // The backward kernel requires bf16 data tensors and gradient outputs, while
  // the log-sum-exp intermediates must remain f32. Verify that the f32 function
  // boundary is preserved with casts around the kernel.
  func.func @sdpa_backward_f32(
      %grad_output: tensor<1x8x64x64xf32>,
      %output: tensor<1x8x64x64xf32>,
      %query: tensor<1x8x64x64xf32>,
      %key: tensor<1x8x64x64xf32>,
      %value: tensor<1x8x64x64xf32>,
      %intermediates: tensor<1x8x64x32xf32>,
      %mask: tensor<1x1x64x64xf32>)
      -> (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>,
          tensor<1x8x64x64xf32>) {
    // CHECK-LABEL: func.func @sdpa_backward_f32
    // CHECK: %[[GRAD_OUTPUT_BF16:.*]] = "ttnn.typecast"(%{{.*}}) : {{.*}} -> tensor<1x8x64x64xbf16
    // CHECK: %[[OUTPUT_BF16:.*]] = "ttnn.typecast"(%{{.*}}) : {{.*}} -> tensor<1x8x64x64xbf16
    // CHECK: %[[QUERY_BF16:.*]] = "ttnn.typecast"(%{{.*}}) : {{.*}} -> tensor<1x8x64x64xbf16
    // CHECK: %[[KEY_BF16:.*]] = "ttnn.typecast"(%{{.*}}) : {{.*}} -> tensor<1x8x64x64xbf16
    // CHECK: %[[VALUE_BF16:.*]] = "ttnn.typecast"(%{{.*}}) : {{.*}} -> tensor<1x8x64x64xbf16
    // CHECK: %[[MASK_BF16:.*]] = "ttnn.typecast"(%{{.*}}) : {{.*}} -> tensor<1x1x64x64xbf16
    // CHECK: %[[GQ_BF16:.*]], %[[GK_BF16:.*]], %[[GV_BF16:.*]] = "ttnn.sdpa_bw"(%[[GRAD_OUTPUT_BF16]], %[[OUTPUT_BF16]], %[[QUERY_BF16]], %[[KEY_BF16]], %[[VALUE_BF16]], %{{.*}}, %[[MASK_BF16]])
    // CHECK-SAME: -> (tensor<1x8x64x64xbf16
    // CHECK-SAME: tensor<1x8x64x64xbf16
    // CHECK-SAME: tensor<1x8x64x64xbf16
    // CHECK-DAG: %[[GQ_F32:.*]] = "ttnn.typecast"(%[[GQ_BF16]]) : {{.*}} -> tensor<1x8x64x64xf32
    // CHECK-DAG: %[[GK_F32:.*]] = "ttnn.typecast"(%[[GK_BF16]]) : {{.*}} -> tensor<1x8x64x64xf32
    // CHECK-DAG: %[[GV_F32:.*]] = "ttnn.typecast"(%[[GV_BF16]]) : {{.*}} -> tensor<1x8x64x64xf32
    // CHECK: return %[[GQ_F32]], %[[GK_F32]], %[[GV_F32]]
    %grad_query, %grad_key, %grad_value = "ttcore.composite"(
        %grad_output, %output, %query, %key, %value, %intermediates, %mask) <{
        composite_name = "sdpa_bw",
        decomposition = @sdpa_bw_f32_decomposition,
        composite_attributes = {
          mask_type = #ttcore.attention_mask_type<arbitrary>,
          dropout_probability = 0.000000e+00 : f32}}>
        : (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>,
           tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>,
           tensor<1x8x64x64xf32>, tensor<1x8x64x32xf32>,
           tensor<1x1x64x64xf32>)
          -> (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>,
              tensor<1x8x64x64xf32>)
    return %grad_query, %grad_key, %grad_value
        : tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>,
          tensor<1x8x64x64xf32>
  }

  // SDPA forward requires interleaved inputs. Verify that a compute-produced
  // query is resharded before reaching the op.
  func.func @sdpa_fw_query_from_l1_producer(
      %q0: tensor<1x8x128x64xbf16>,
      %q1: tensor<1x8x128x64xbf16>,
      %key: tensor<1x8x128x64xbf16>,
      %value: tensor<1x8x128x64xbf16>)
      -> tensor<1x8x128x64xbf16> {
    // The query reaches sdpa_fw through a single reshard: the multiply runs in
    // its own layout and one to_memory_config converts straight to the
    // interleaved layout sdpa_fw requires. This used to take two chained
    // to_memory_config ops; the intermediate hop is now folded away.
    // CHECK-LABEL: func.func @sdpa_fw_query_from_l1_producer
    // CHECK: %[[MUL:.*]] = "ttnn.multiply"
    // CHECK: %[[RESHARD:.*]] = "ttnn.to_memory_config"(%[[MUL]])
    // CHECK: "ttnn.sdpa_fw"(%[[RESHARD]],
    %query = "ttir.multiply"(%q0, %q1)
        : (tensor<1x8x128x64xbf16>, tensor<1x8x128x64xbf16>)
          -> tensor<1x8x128x64xbf16>
    %output = "ttcore.composite"(%query, %key, %value) <{
        composite_name = "sdpa_fw",
        decomposition = @sdpa_fw_128_decomposition,
        composite_attributes = {
          mask_type = #ttcore.attention_mask_type<causal>,
          dropout_probability = 0.000000e+00 : f32,
          return_intermediates = false}}>
        : (tensor<1x8x128x64xbf16>, tensor<1x8x128x64xbf16>,
           tensor<1x8x128x64xbf16>)
          -> tensor<1x8x128x64xbf16>
    return %output : tensor<1x8x128x64xbf16>
  }

  func.func private @sdpa_fw_bf16_decomposition(
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>, %mask: tensor<1x1x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x32xf32>) {
    %intermediates = "ttir.empty"() : () -> tensor<1x8x64x32xf32>
    return %query, %intermediates : tensor<1x8x64x64xbf16>,
        tensor<1x8x64x32xf32>
  }

  func.func private @sdpa_bw_bf16_decomposition(
      %grad_output: tensor<1x8x64x64xbf16>,
      %attn_output: tensor<1x8x64x64xbf16>,
      %query: tensor<1x8x64x64xbf16>, %key: tensor<1x8x64x64xbf16>,
      %value: tensor<1x8x64x64xbf16>,
      %intermediates: tensor<1x8x64x32xf32>,
      %mask: tensor<1x1x64x64xbf16>)
      -> (tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>,
          tensor<1x8x64x64xbf16>) {
    return %query, %key, %value : tensor<1x8x64x64xbf16>,
        tensor<1x8x64x64xbf16>, tensor<1x8x64x64xbf16>
  }

  func.func private @sdpa_fw_f32_decomposition(
      %query: tensor<1x8x64x64xf32>, %key: tensor<1x8x64x64xf32>,
      %value: tensor<1x8x64x64xf32>, %mask: tensor<1x1x64x64xf32>)
      -> (tensor<1x8x64x64xf32>, tensor<1x8x64x32xf32>) {
    %intermediates = "ttir.empty"() : () -> tensor<1x8x64x32xf32>
    return %query, %intermediates : tensor<1x8x64x64xf32>,
        tensor<1x8x64x32xf32>
  }

  func.func private @sdpa_bw_f32_decomposition(
      %grad_output: tensor<1x8x64x64xf32>,
      %attn_output: tensor<1x8x64x64xf32>,
      %query: tensor<1x8x64x64xf32>, %key: tensor<1x8x64x64xf32>,
      %value: tensor<1x8x64x64xf32>,
      %intermediates: tensor<1x8x64x32xf32>,
      %mask: tensor<1x1x64x64xf32>)
      -> (tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>,
          tensor<1x8x64x64xf32>) {
    return %query, %key, %value : tensor<1x8x64x64xf32>,
        tensor<1x8x64x64xf32>, tensor<1x8x64x64xf32>
  }

  func.func private @sdpa_fw_128_decomposition(
      %query: tensor<1x8x128x64xbf16>, %key: tensor<1x8x128x64xbf16>,
      %value: tensor<1x8x128x64xbf16>) -> tensor<1x8x128x64xbf16> {
    return %query : tensor<1x8x128x64xbf16>
  }
}
