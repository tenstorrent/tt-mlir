// RUN: ttmlir-opt --ttir-to-ttnn-runtime-pipeline -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // The logits stay tiled bf16, but ttml::metal::cross_entropy_fw reads the class
  // indices as row-major uint32, so the operand workarounds must insert a
  // to_layout that untilizes target.
  func.func @cross_entropy_fw(%input: tensor<4x1x32x64xbf16>, %target: tensor<4x32xui32>)
      -> tensor<4x1x32x1xbf16> {
    // CHECK-DAG: #[[INPUT_TILED:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x2x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[TARGET_RM:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x32xui32, #dram>, <interleaved>>

    // The target arrives tiled and gets untilized to row-major ui32.
    // CHECK: %[[TARGET:[0-9]+]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: -> tensor<4x32xui32, #[[TARGET_RM]]>

    // CHECK: "ttnn.cross_entropy_fw"(%arg0, %[[TARGET]])
    // CHECK-SAME: (tensor<4x1x32x64xbf16, #[[INPUT_TILED]]>, tensor<4x32xui32, #[[TARGET_RM]]>)
    // CHECK-SAME: -> tensor<4x1x32x1xbf16
    %0 = "ttir.cross_entropy_fw"(%input, %target)
        : (tensor<4x1x32x64xbf16>, tensor<4x32xui32>) -> tensor<4x1x32x1xbf16>
    return %0 : tensor<4x1x32x1xbf16>
  }
}
