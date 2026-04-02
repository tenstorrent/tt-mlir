// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-workaround="ttnn-is-optimizer-enabled=true" --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#tile_bf16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#tile_i32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32 + d1 * 32 + d2, d3), <1x1>, memref<1x1x!ttcore.tile<32x32, si32>, #dram>, <interleaved>>

module attributes {} {
  func.func @scatter_optimizer_enabled(%arg0: tensor<1x1x32x32xbf16, #tile_bf16>, %arg1: tensor<1x1x32x32xsi32, #tile_i32>, %arg2: tensor<1x1x32x32xbf16, #tile_bf16>) -> tensor<1x1x32x32xbf16, #tile_bf16> {
    // CHECK-LABEL: func.func @scatter_optimizer_enabled
    // CHECK: %[[INPUT_RM:.*]] = "ttnn.to_layout"(%arg0)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: -> tensor<1x1x32x32xbf16
    // CHECK: %[[INDEX_RM:.*]] = "ttnn.to_layout"(%arg1)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: -> tensor<1x1x32x32xsi32
    // CHECK: %[[SOURCE_RM:.*]] = "ttnn.to_layout"(%arg2)
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-SAME: -> tensor<1x1x32x32xbf16
    // CHECK: %[[SCATTER:.*]] = "ttnn.scatter"(%[[INPUT_RM]], %[[INDEX_RM]], %[[SOURCE_RM]])
    // CHECK-SAME: <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}>
    // CHECK: %[[RESTORE:.*]] = "ttnn.to_layout"(%[[SCATTER]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK: return %[[RESTORE]]
    %0 = "ttnn.scatter"(%arg0, %arg1, %arg2) <{dim = 0 : i32, scatter_reduce_type = #ttcore.reduce_type<invalid>}> : (tensor<1x1x32x32xbf16, #tile_bf16>, tensor<1x1x32x32xsi32, #tile_i32>, tensor<1x1x32x32xbf16, #tile_bf16>) -> tensor<1x1x32x32xbf16, #tile_bf16>
    %1 = "ttnn.to_layout"(%0) <{layout = #ttnn.layout<tile>}> : (tensor<1x1x32x32xbf16, #tile_bf16>) -> tensor<1x1x32x32xbf16, #tile_bf16>
    return %1 : tensor<1x1x32x32xbf16, #tile_bf16>
  }
}
