// RUN: ttmlir-opt --ttnn-memory-management -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#layout_128x128 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_64x64 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_32x128 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_128x32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_32x32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_1x1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1xf32, #dram>, <interleaved>>
#layout_32x64 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_64x32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_r_1x1x1x2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<2xf32, #dram>, <interleaved>>
#layout_r_8x1x1x2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<16xf32, #dram>, <interleaved>>
#layout_r_1x1x1x8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<8xf32, #dram>, <interleaved>>
#layout_r_1x1x8x2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<16xf32, #dram>, <interleaved>>
#layout_r_1x1x1x16 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<16xf32, #dram>, <interleaved>>
#layout_r_1x1x1x1024 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1024xf32, #dram>, <interleaved>>
#layout_r_1x1x32x32 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0, d1, d2, d3), <1x1>, memref<1024xf32, #dram>, <interleaved>>
#layout_mm_1x8190x6x3072 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 262080 + d1 * 32 + d2, d3), <1x1>, memref<8190x96x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_mm_1x8190x3072 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 8192 + d1, d2), <1x1>, memref<256x96x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_mm_1x49140x3072 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 49152 + d1, d2), <1x1>, memref<1536x96x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  // sliceReshape
  // CHECK: %[[SLICE:.*]] = "ttnn.slice_static"(%arg0) <{begins = [32 : i32, 0 : i32], ends = [64 : i32, 128 : i32], step = [1 : i32, 1 : i32]}>
  // CHECK-NOT: "ttnn.reshape"
  // CHECK: return %[[SLICE]]
  func.func @slice_reshape(%arg0: tensor<128x128xf32, #layout_128x128>) -> tensor<32x128xf32, #layout_32x128> {
    %0 = "ttnn.reshape"(%arg0) <{shape = [128 : i32, 128 : i32]}> : (tensor<128x128xf32, #layout_128x128>) -> tensor<128x128xf32, #layout_128x128>
    %1 = "ttnn.slice_static"(%0) <{begins = [32 : i32, 0 : i32], ends = [64 : i32, 128 : i32], step = [1 : i32, 1 : i32]}> : (tensor<128x128xf32, #layout_128x128>) -> tensor<32x128xf32, #layout_32x128>
    return %1 : tensor<32x128xf32, #layout_32x128>
  }

  // slicePermute
  // CHECK: %[[SLICE:.*]] = "ttnn.slice_static"(%arg0) <{begins = [32 : i32, 0 : i32], ends = [64 : i32, 128 : i32], step = [1 : i32, 1 : i32]}>
  // CHECK: %[[PERM:.*]] = "ttnn.permute"(%[[SLICE]])
  // CHECK-SAME: permutation = array<i64: 1, 0>
  // CHECK-NOT: "ttnn.slice_static"(%[[PERM]])
  func.func @slice_permute(%arg0: tensor<128x128xf32, #layout_128x128>) -> tensor<128x32xf32, #layout_128x32> {
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 1, 0>}> : (tensor<128x128xf32, #layout_128x128>) -> tensor<128x128xf32, #layout_128x128>
    %1 = "ttnn.slice_static"(%0) <{begins = [0 : i32, 32 : i32], ends = [128 : i32, 64 : i32], step = [1 : i32, 1 : i32]}> : (tensor<128x128xf32, #layout_128x128>) -> tensor<128x32xf32, #layout_128x32>
    return %1 : tensor<128x32xf32, #layout_128x32>
  }

  // sliceEltwise
  // CHECK: %[[LHS_SLICE:.*]] = "ttnn.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 32 : i32], step = [1 : i32, 1 : i32]}>
  // CHECK: %[[RHS_SLICE:.*]] = "ttnn.slice_static"(%arg1) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 32 : i32], step = [1 : i32, 1 : i32]}>
  // CHECK: %[[ADD:.*]] = "ttnn.add"(%[[LHS_SLICE]], %[[RHS_SLICE]]) <{dtype = #ttcore.supportedDataTypes<f32>}>
  // CHECK-NOT: "ttnn.slice_static"(%[[ADD]])
  func.func @slice_eltwise(%arg0: tensor<64x64xf32, #layout_64x64>, %arg1: tensor<64x64xf32, #layout_64x64>) -> tensor<32x32xf32, #layout_32x32> {
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<64x64xf32, #layout_64x64>, tensor<64x64xf32, #layout_64x64>) -> tensor<64x64xf32, #layout_64x64>
    %1 = "ttnn.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 32 : i32], step = [1 : i32, 1 : i32]}> : (tensor<64x64xf32, #layout_64x64>) -> tensor<32x32xf32, #layout_32x32>
    return %1 : tensor<32x32xf32, #layout_32x32>
  }

  // sliceEltwiseWithBroadcast
  // CHECK: %[[LHS_SLICE:.*]] = "ttnn.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 32 : i32], step = [1 : i32, 1 : i32]}>
  // CHECK: %[[ADD:.*]] = "ttnn.add"(%[[LHS_SLICE]], %arg1) <{dtype = #ttcore.supportedDataTypes<f32>}>
  // CHECK-NOT: "ttnn.slice_static"(%[[ADD]])
  func.func @slice_eltwise_with_broadcast(%arg0: tensor<64x64xf32, #layout_64x64>, %arg1: tensor<1x1xf32, #layout_1x1>) -> tensor<32x32xf32, #layout_32x32> {
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<64x64xf32, #layout_64x64>, tensor<1x1xf32, #layout_1x1>) -> tensor<64x64xf32, #layout_64x64>
    %1 = "ttnn.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 32 : i32], step = [1 : i32, 1 : i32]}> : (tensor<64x64xf32, #layout_64x64>) -> tensor<32x32xf32, #layout_32x32>
    return %1 : tensor<32x32xf32, #layout_32x32>
  }

  // sliceRepeat
  // CHECK: %[[SLICE:.*]] = "ttnn.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 32 : i32], step = [1 : i32, 1 : i32]}>
  // CHECK: %[[REPEAT:.*]] = "ttnn.repeat"(%[[SLICE]]) <{repeat_dims = #ttnn.shape<2x1>}>
  // CHECK-NOT: "ttnn.slice_static"(%[[REPEAT]])
  func.func @slice_repeat(%arg0: tensor<32x64xf32, #layout_32x64>) -> tensor<64x32xf32, #layout_64x32> {
    %0 = "ttnn.repeat"(%arg0) <{repeat_dims = #ttnn.shape<2x1>}> : (tensor<32x64xf32, #layout_32x64>) -> tensor<64x64xf32, #layout_64x64>
    %1 = "ttnn.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [64 : i32, 32 : i32], step = [1 : i32, 1 : i32]}> : (tensor<64x64xf32, #layout_64x64>) -> tensor<64x32xf32, #layout_64x32>
    return %1 : tensor<64x32xf32, #layout_64x32>
  }

  // repeatReshape
  // CHECK: %[[REPEAT:.*]] = "ttnn.repeat"(%arg0) <{repeat_dims = #ttnn.shape<1x1x8x1>}>
  // CHECK-NOT: repeat_dims = #ttnn.shape<8x1x1x1>
  // CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"(%[[REPEAT]]) <{shape = [1 : i32, 1 : i32, 1 : i32, 16 : i32]}>
  func.func @repeat_reshape(%arg0: tensor<1x1x1x2xf32, #layout_r_1x1x1x2>) -> tensor<1x1x1x16xf32, #layout_r_1x1x1x16> {
    %0 = "ttnn.repeat"(%arg0) <{repeat_dims = #ttnn.shape<8x1x1x1>}> : (tensor<1x1x1x2xf32, #layout_r_1x1x1x2>) -> tensor<8x1x1x2xf32, #layout_r_8x1x1x2>
    %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 1 : i32, 1 : i32, 16 : i32]}> : (tensor<8x1x1x2xf32, #layout_r_8x1x1x2>) -> tensor<1x1x1x16xf32, #layout_r_1x1x1x16>
    return %1 : tensor<1x1x1x16xf32, #layout_r_1x1x1x16>
  }

  // reshape-eltwise adjust
  // CHECK: %[[R0:.*]] = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 1 : i32, 32 : i32, 32 : i32]}>
  // CHECK: %[[R1:.*]] = "ttnn.reshape"(%arg1) <{shape = [1 : i32, 1 : i32, 32 : i32, 32 : i32]}>
  // CHECK: %[[ADD:.*]] = "ttnn.add"(%[[R0]], %[[R1]]) <{dtype = #ttcore.supportedDataTypes<f32>}>
  // CHECK-NOT: "ttnn.reshape"(%[[ADD]])
  func.func @reshape_eltwise(%arg0: tensor<1x1x1x1024xf32, #layout_r_1x1x1x1024>, %arg1: tensor<1x1x1x1024xf32, #layout_r_1x1x1x1024>) -> tensor<1x1x32x32xf32, #layout_r_1x1x32x32> {
    %0 = "ttnn.add"(%arg0, %arg1) <{dtype = #ttcore.supportedDataTypes<f32>}> : (tensor<1x1x1x1024xf32, #layout_r_1x1x1x1024>, tensor<1x1x1x1024xf32, #layout_r_1x1x1x1024>) -> tensor<1x1x1x1024xf32, #layout_r_1x1x1x1024>
    %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 1 : i32, 32 : i32, 32 : i32]}> : (tensor<1x1x1x1024xf32, #layout_r_1x1x1x1024>) -> tensor<1x1x32x32xf32, #layout_r_1x1x32x32>
    return %1 : tensor<1x1x32x32xf32, #layout_r_1x1x32x32>
  }

  // hoist common reshape above sibling slices
  // CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"(%arg0) <{shape = [1 : i32, 49140 : i32, 3072 : i32]}>
  // CHECK-DAG: %[[SLICE0:.*]] = "ttnn.slice_static"(%[[RESHAPE]]) <{begins = [0 : i32, 2 : i32, 0 : i32], ends = [1 : i32, 49140 : i32, 3072 : i32], step = [1 : i32, 6 : i32, 1 : i32]}>
  // CHECK-DAG: %[[SLICE1:.*]] = "ttnn.slice_static"(%[[RESHAPE]]) <{begins = [0 : i32, 3 : i32, 0 : i32], ends = [1 : i32, 49140 : i32, 3072 : i32], step = [1 : i32, 6 : i32, 1 : i32]}>
  // CHECK: return %[[SLICE0]], %[[SLICE1]]
  func.func @hoist_common_reshape_above_slices(%arg0: tensor<1x8190x6x3072xf32, #layout_mm_1x8190x6x3072>) -> (tensor<1x8190x3072xf32, #layout_mm_1x8190x3072>, tensor<1x8190x3072xf32, #layout_mm_1x8190x3072>) {
    %0 = "ttnn.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 2 : i32, 0 : i32], ends = [1 : i32, 8190 : i32, 3 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8190x6x3072xf32, #layout_mm_1x8190x6x3072>) -> tensor<1x8190x1x3072xf32, #layout_mm_1x8190x6x3072>
    %1 = "ttnn.reshape"(%0) <{shape = [1 : i32, 8190 : i32, 3072 : i32]}> : (tensor<1x8190x1x3072xf32, #layout_mm_1x8190x6x3072>) -> tensor<1x8190x3072xf32, #layout_mm_1x8190x3072>
    %2 = "ttnn.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32, 3 : i32, 0 : i32], ends = [1 : i32, 8190 : i32, 4 : i32, 3072 : i32], step = [1 : i32, 1 : i32, 1 : i32, 1 : i32]}> : (tensor<1x8190x6x3072xf32, #layout_mm_1x8190x6x3072>) -> tensor<1x8190x1x3072xf32, #layout_mm_1x8190x6x3072>
    %3 = "ttnn.reshape"(%2) <{shape = [1 : i32, 8190 : i32, 3072 : i32]}> : (tensor<1x8190x1x3072xf32, #layout_mm_1x8190x6x3072>) -> tensor<1x8190x3072xf32, #layout_mm_1x8190x3072>
    return %1, %3 : tensor<1x8190x3072xf32, #layout_mm_1x8190x3072>, tensor<1x8190x3072xf32, #layout_mm_1x8190x3072>
  }
}
