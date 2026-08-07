// RUN: ttmlir-opt --split-input-file --ttir-to-ttnn-runtime-pipeline -o %t %s
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
    %0 = "ttir.cross_entropy_fw"(%input, %target)
        : (tensor<4x1x32x64xf32>, tensor<4x32xui32>) -> tensor<4x1x32x1xf32>
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
    %0 = "ttir.cross_entropy_fw"(%input, %target)
        : (tensor<32x64xbf16>, tensor<32xui32>) -> tensor<32x1xbf16>
    return %0 : tensor<32x1xbf16>
  }
}
