// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-runtime-pipeline="composite-resolution=force-promote" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // The logits stay tiled bf16, but ttml::metal::cross_entropy_fw reads the class
  // indices as row-major uint32, so the operand workarounds must insert a
  // to_layout that untilizes target.

  // CHECK-DAG: #[[INPUT_TILED:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
  // CHECK-DAG: #[[TARGET_RM:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x32xui32, #dram>, <interleaved>>
  // CHECK-LABEL: func.func @cross_entropy_fw
  func.func @cross_entropy_fw(%input: tensor<4x1x32x64xf32>, %target: tensor<4x32xui32>)
      -> tensor<4x1x32x1xf32> {
    // The f32 logits are typecast to bf16, staying tiled.
    // CHECK: %[[INPUT:[0-9]+]] = "ttnn.typecast"(%arg0)
    // CHECK-SAME: -> tensor<4x1x32x64xbf16, #[[INPUT_TILED]]>

    // The target arrives tiled and gets untilized to row-major ui32.
    // CHECK: %[[TARGET:[0-9]+]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: -> tensor<4x32xui32, #[[TARGET_RM]]>

    // CHECK: "ttnn.cross_entropy_fw"(%[[INPUT]], %[[TARGET]])
    // CHECK-SAME: (tensor<4x1x32x64xbf16, #[[INPUT_TILED]]>, tensor<4x32xui32, #[[TARGET_RM]]>)
    // CHECK-SAME: -> tensor<4x1x32x1xbf16
    %0 = "ttcore.composite"(%input, %target) <{
      composite_name = "cross_entropy_fw", decomposition = @decomposition}>
        : (tensor<4x1x32x64xf32>, tensor<4x32xui32>) -> tensor<4x1x32x1xf32>
    return %0 : tensor<4x1x32x1xf32>
  }
  func.func private @decomposition(
      %input: tensor<4x1x32x64xf32>,
      %target: tensor<4x32xui32>) -> tensor<4x1x32x1xf32> {
    %0 = "ttir.zeros"() <{shape = array<i32: 4, 1, 32, 1>}> : () -> tensor<4x1x32x1xf32>
    return %0 : tensor<4x1x32x1xf32>
  }
}

// -----

// A rank-2 input reaches the kernel as (1, 1, H, W) with a (1, H) target. This
// exercises the decomposition through the real pipeline.
// CHECK-LABEL: func.func @rank2_input
module {
  func.func @rank2_input(%input: tensor<32x64xbf16>, %target: tensor<32xui32>)
      -> tensor<32x1xbf16> {
    // CHECK: "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 32 : i32, 64 : i32]}>
    // CHECK: "ttnn.reshape"(%arg1) <{shape = [1 : i32, 32 : i32]}>
    // CHECK: "ttnn.cross_entropy_fw"
    // CHECK-SAME: -> tensor<1x1x32x1xbf16
    // CHECK: "ttnn.reshape"({{.*}}) <{shape = [32 : i32, 1 : i32]}>
    %0 = "ttcore.composite"(%input, %target) <{
      composite_name = "cross_entropy_fw", decomposition = @decomposition}>
        : (tensor<32x64xbf16>, tensor<32xui32>) -> tensor<32x1xbf16>
    return %0 : tensor<32x1xbf16>
  }
  func.func private @decomposition(
      %input: tensor<32x64xbf16>,
      %target: tensor<32xui32>) -> tensor<32x1xbf16> {
    %0 = "ttir.zeros"() <{shape = array<i32: 32, 1>}> : () -> tensor<32x1xbf16>
    return %0 : tensor<32x1xbf16>
  }
}

// -----

// ttml::metal::cross_entropy_fw asserts that the input, target and output
// buffers all live in DRAM, so L1-interleaved operands must be moved out.

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
#input_l1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<2x3x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#target_l1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x32xui32, #l1>, <interleaved>>
#output_l1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #l1>, <interleaved>>
#output_dram = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

// CHECK-DAG: #[[INPUT_DRAM:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<2x3x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// CHECK-DAG: #[[TARGET_DRAM:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<2x32xui32, #dram>, <interleaved>>
// CHECK-DAG: #[[OUTPUT_DRAM:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<2x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// CHECK-LABEL: func.func @l1_interleaved_operands
module {
  func.func @l1_interleaved_operands(%input: tensor<2x1x32x96xbf16, #input_l1>, %target: tensor<2x32xui32, #target_l1>)
      -> tensor<2x1x32x1xbf16, #output_dram> {
    // CHECK: %[[INPUT:[0-9]+]] = "ttnn.to_memory_config"(%arg0)
    // CHECK-SAME: -> tensor<2x1x32x96xbf16, #[[INPUT_DRAM]]>
    // CHECK: %[[TARGET:[0-9]+]] = "ttnn.to_memory_config"(%arg1)
    // CHECK-SAME: -> tensor<2x32xui32, #[[TARGET_DRAM]]>
    // CHECK: "ttnn.cross_entropy_fw"(%[[INPUT]], %[[TARGET]])
    // CHECK-SAME: -> tensor<2x1x32x1xbf16, #[[OUTPUT_DRAM]]>
    %0 = "ttnn.cross_entropy_fw"(%input, %target)
        : (tensor<2x1x32x96xbf16, #input_l1>, tensor<2x32xui32, #target_l1>) -> tensor<2x1x32x1xbf16, #output_l1>
    %1 = "ttnn.to_memory_config"(%0) : (tensor<2x1x32x1xbf16, #output_l1>) -> tensor<2x1x32x1xbf16, #output_dram>
    return %1 : tensor<2x1x32x1xbf16, #output_dram>
  }
}
