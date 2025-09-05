// REQUIRES: opmodel
// RUN: ttmlir-opt --ttcore-register-device --ttnn-operation-validation-and-fallback %s -o %t.mlir
// RUN: FileCheck %s --input-file %t.mlir

#dram = #ttnn.buffer_type<dram>
#ttnn_layout_tile_si32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>
#ttnn_layout_tile_f32 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<32x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module attributes {} {
  func.func @reshape_with_int32(%arg0: tensor<1x1x32x32xsi32, #ttnn_layout_tile_si32>) -> tensor<1x32x32xf32, #ttnn_layout_tile_f32> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    // CHECK: %[[RESHAPE_RES:.*]] = "ttnn.reshape"
    // CHECK: "ttnn.to_layout"
    // CHECK-SAME: (%[[RESHAPE_RES]]

    %1 = "ttnn.reshape"(%arg0) <{
      shape = [1 : i32, 32 : i32, 32 : i32]
    }> : (tensor<1x1x32x32xsi32, #ttnn_layout_tile_si32>) -> tensor<1x32x32xf32, #ttnn_layout_tile_f32>

    return %1 : tensor<1x32x32xf32, #ttnn_layout_tile_f32>
  }
}
