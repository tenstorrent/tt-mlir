// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t
#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#bf16_l1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#f32_l1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>, <interleaved>>
#f16_l1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f16>, #l1>, <interleaved>>
#si32_l1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #l1>, <interleaved>>
#bf16_dram = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#f32_dram = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module attributes {} {
  // Positive: bf16 -> f32 (typecast) -> bf16 (to_layout, + memcfg change) is a
  // lossless round-trip. The to_layout reads %arg0 directly (becoming a pure
  // layout change) and the now-dead typecast is removed by DCE.
  // CHECK-LABEL: func.func @merge_lossless_roundtrip
  func.func @merge_lossless_roundtrip(%arg0: tensor<32x32xbf16, #bf16_l1>) -> tensor<32x32xbf16, #bf16_dram> {
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: "ttnn.to_tensor_spec"(%arg0)
    // CHECK-SAME: -> tensor<32x32xbf16,
    %0 = "ttnn.typecast"(%arg0) : (tensor<32x32xbf16, #bf16_l1>) -> tensor<32x32xf32, #f32_l1>
    %1 = "ttnn.to_tensor_spec"(%0) : (tensor<32x32xf32, #f32_l1>) -> tensor<32x32xbf16, #bf16_dram>
    return %1 : tensor<32x32xbf16, #bf16_dram>
  }

  // Positive, multi-use: the typecast also feeds another consumer (the f32
  // result), so it must survive; only the to_layout is rewired to read %arg0.
  // CHECK-LABEL: func.func @merge_lossless_roundtrip_multi_use
  func.func @merge_lossless_roundtrip_multi_use(%arg0: tensor<32x32xbf16, #bf16_l1>) -> (tensor<32x32xbf16, #bf16_dram>, tensor<32x32xf32, #f32_l1>) {
    // CHECK: "ttnn.typecast"(%arg0)
    // CHECK: "ttnn.to_tensor_spec"(%arg0)
    // CHECK-SAME: -> tensor<32x32xbf16,
    %0 = "ttnn.typecast"(%arg0) : (tensor<32x32xbf16, #bf16_l1>) -> tensor<32x32xf32, #f32_l1>
    %1 = "ttnn.to_tensor_spec"(%0) : (tensor<32x32xf32, #f32_l1>) -> tensor<32x32xbf16, #bf16_dram>
    return %1, %0 : tensor<32x32xbf16, #bf16_dram>, tensor<32x32xf32, #f32_l1>
  }

  // Negative: f32 -> bf16 -> f32 is a round-trip but the producer is lossy
  // (precision reduction at bf16). Must NOT merge.
  // CHECK-LABEL: func.func @no_merge_lossy_bf16_roundtrip
  func.func @no_merge_lossy_bf16_roundtrip(%arg0: tensor<32x32xf32, #f32_l1>) -> tensor<32x32xf32, #f32_dram> {
    // CHECK: %[[TC:.*]] = "ttnn.typecast"(%arg0)
    // CHECK: "ttnn.to_tensor_spec"(%[[TC]])
    %0 = "ttnn.typecast"(%arg0) : (tensor<32x32xf32, #f32_l1>) -> tensor<32x32xbf16, #bf16_l1>
    %1 = "ttnn.to_tensor_spec"(%0) : (tensor<32x32xbf16, #bf16_l1>) -> tensor<32x32xf32, #f32_dram>
    return %1 : tensor<32x32xf32, #f32_dram>
  }

  // Negative: bf16 -> f16 -> bf16 is a round-trip but the producer is lossy
  // (f16 has fewer exponent bits than bf16). Must NOT merge.
  // CHECK-LABEL: func.func @no_merge_lossy_f16_roundtrip
  func.func @no_merge_lossy_f16_roundtrip(%arg0: tensor<32x32xbf16, #bf16_l1>) -> tensor<32x32xbf16, #bf16_dram> {
    // CHECK: %[[TC:.*]] = "ttnn.typecast"(%arg0)
    // CHECK: "ttnn.to_tensor_spec"(%[[TC]])
    %0 = "ttnn.typecast"(%arg0) : (tensor<32x32xbf16, #bf16_l1>) -> tensor<32x32xf16, #f16_l1>
    %1 = "ttnn.to_tensor_spec"(%0) : (tensor<32x32xf16, #f16_l1>) -> tensor<32x32xbf16, #bf16_dram>
    return %1 : tensor<32x32xbf16, #bf16_dram>
  }

  // Negative: f32 -> si32 -> f32 (FP->Int->FP quantization). Producer is lossy
  // (float to int). Must NOT merge.
  // CHECK-LABEL: func.func @no_merge_fp_int_fp
  func.func @no_merge_fp_int_fp(%arg0: tensor<32x32xf32, #f32_l1>) -> tensor<32x32xf32, #f32_dram> {
    // CHECK: %[[TC:.*]] = "ttnn.typecast"(%arg0)
    // CHECK: "ttnn.to_tensor_spec"(%[[TC]])
    %0 = "ttnn.typecast"(%arg0) : (tensor<32x32xf32, #f32_l1>) -> tensor<32x32xsi32, #si32_l1>
    %1 = "ttnn.to_tensor_spec"(%0) : (tensor<32x32xsi32, #si32_l1>) -> tensor<32x32xf32, #f32_dram>
    return %1 : tensor<32x32xf32, #f32_dram>
  }
}
