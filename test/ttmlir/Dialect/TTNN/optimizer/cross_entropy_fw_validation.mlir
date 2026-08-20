// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // The target enters the pipeline tiled, but the TTML kernel requires
  // row-major uint32. The kernel does not accept an output memory-config hint;
  // OpModel derives the tiled, DRAM-interleaved output layout from the input.
  func.func @cross_entropy_fw(
      %input: tensor<4x1x32x64xbf16>,
      %target: tensor<4x32xui32>) -> tensor<4x1x32x1xbf16> {
    // CHECK-DAG: #[[INPUT_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[TARGET_TILED_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
    // CHECK-DAG: #[[OUTPUT_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[TARGET_RM_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x32xui32, #dram>, <interleaved>>
    // CHECK-DAG: #[[INPUT_F32_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
    // CHECK-DAG: #[[TARGET_I32_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
    // CHECK-DAG: #[[OUTPUT_F32_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

    // CHECK-LABEL: func.func @cross_entropy_fw(
    // CHECK-SAME: %[[INPUT:[0-9a-z_]+]]: tensor<4x1x32x64xbf16, #[[INPUT_LAYOUT]]>
    // CHECK-SAME: %[[TARGET:[0-9a-z_]+]]: tensor<4x32xui32, #[[TARGET_TILED_LAYOUT]]>)
    // CHECK-SAME: -> tensor<4x1x32x1xbf16, #[[OUTPUT_LAYOUT]]>
    // CHECK: %[[TARGET_RM:[0-9a-z_]+]] = "ttnn.to_layout"(%[[TARGET]])
    // CHECK-SAME: -> tensor<4x32xui32, #[[TARGET_RM_LAYOUT]]>
    // CHECK: %[[LOSS:[0-9a-z_]+]] = "ttnn.cross_entropy_fw"(%[[INPUT]], %[[TARGET_RM]])
    // CHECK-SAME: -> tensor<4x1x32x1xbf16, #[[OUTPUT_LAYOUT]]>
    // CHECK: return %[[LOSS]]
    %loss = "ttir.cross_entropy_fw"(%input, %target)
        : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>)
          -> tensor<4x1x32x1xbf16>
    return %loss : tensor<4x1x32x1xbf16>
  }

  // Exercise the operand workarounds together with OpModel validation: the
  // kernel requires bf16 logits and row-major uint32 targets even when the
  // function boundary uses f32 and signed i32.
  func.func @cross_entropy_fw_wrong_types(
      %input: tensor<4x1x32x64xf32>,
      %target: tensor<4x32xi32>) -> tensor<4x1x32x1xf32> {
    // CHECK-LABEL: func.func @cross_entropy_fw_wrong_types(
    // CHECK-SAME: %[[INPUT_F32:[0-9a-z_]+]]: tensor<4x1x32x64xf32, #[[INPUT_F32_LAYOUT]]>
    // CHECK-SAME: %[[TARGET_I32:[0-9a-z_]+]]: tensor<4x32xsi32, #[[TARGET_I32_LAYOUT]]>)
    // CHECK-SAME: -> tensor<4x1x32x1xf32, #[[OUTPUT_F32_LAYOUT]]>
    // CHECK: %[[INPUT_BF16:[0-9a-z_]+]] = "ttnn.typecast"(%[[INPUT_F32]])
    // CHECK-SAME: -> tensor<4x1x32x64xbf16, #[[INPUT_LAYOUT]]>
    // CHECK: %[[TARGET_U32:[0-9a-z_]+]] = "ttnn.typecast"(%[[TARGET_I32]])
    // CHECK-SAME: -> tensor<4x32xui32, #[[TARGET_TILED_LAYOUT]]>
    // CHECK: %[[TARGET_U32_RM:[0-9a-z_]+]] = "ttnn.to_layout"(%[[TARGET_U32]])
    // CHECK-SAME: -> tensor<4x32xui32, #[[TARGET_RM_LAYOUT]]>
    // CHECK: %[[LOSS_BF16:[0-9a-z_]+]] = "ttnn.cross_entropy_fw"(%[[INPUT_BF16]], %[[TARGET_U32_RM]])
    // CHECK-SAME: -> tensor<4x1x32x1xbf16, #[[OUTPUT_LAYOUT]]>
    // CHECK: %[[LOSS_F32:[0-9a-z_]+]] = "ttnn.typecast"(%[[LOSS_BF16]])
    // CHECK-SAME: -> tensor<4x1x32x1xf32, #[[OUTPUT_F32_LAYOUT]]>
    // CHECK: return %[[LOSS_F32]]
    %loss = "ttir.cross_entropy_fw"(%input, %target)
        : (tensor<4x1x32x64xf32>, tensor<4x32xi32>)
          -> tensor<4x1x32x1xf32>
    return %loss : tensor<4x1x32x1xf32>
  }
}
