// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-memory-management -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttnn.buffer_type<dram>
#l1 = #ttnn.buffer_type<l1>
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
#layout_1x67M_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x2097152x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_67Mx1_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2097152x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_67Mx2_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2097152x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_8192x8192_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<256x256x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_8192x16384_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<256x512x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_4d_1x1x8192x8192_tile = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8192 + d1 * 8192 + d2, d3), <1x1>, memref<256x256x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_4d_8192x8192x1x1_tile = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 8192 + d1 + d2, d3), <1x1>, memref<2097152x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_E_F = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<3387x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_1d_t = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x216730x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_1d_rm = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x6935360xf32, #dram>, <interleaved>>
#layout_64x64_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x64xf32, #dram>, <interleaved>>
#layout_1d_rm_l1 = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x6935360xf32, #l1>, <interleaved>>
#layout_E_F_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<108365x64xf32, #dram>, <interleaved>>
#layout_100x50 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x2x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_5000_t = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x157x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#layout_5000_rm = #ttnn.ttnn_layout<(d0) -> (0, d0), <1x1>, memref<1x5000xf32, #dram>, <interleaved>>

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
  // CHECK: %[[ADD:.*]] = "ttnn.add"(%[[LHS_SLICE]], %[[RHS_SLICE]])
  // CHECK-NOT: "ttnn.slice_static"(%[[ADD]])
  func.func @slice_eltwise(%arg0: tensor<64x64xf32, #layout_64x64>, %arg1: tensor<64x64xf32, #layout_64x64>) -> tensor<32x32xf32, #layout_32x32> {
    %0 = "ttnn.add"(%arg0, %arg1) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x64xf32, #layout_64x64>, tensor<64x64xf32, #layout_64x64>) -> tensor<64x64xf32, #layout_64x64>
    %1 = "ttnn.slice_static"(%0) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 32 : i32], step = [1 : i32, 1 : i32]}> : (tensor<64x64xf32, #layout_64x64>) -> tensor<32x32xf32, #layout_32x32>
    return %1 : tensor<32x32xf32, #layout_32x32>
  }

  // sliceEltwiseWithBroadcast
  // CHECK: %[[LHS_SLICE:.*]] = "ttnn.slice_static"(%arg0) <{begins = [0 : i32, 0 : i32], ends = [32 : i32, 32 : i32], step = [1 : i32, 1 : i32]}>
  // CHECK: %[[ADD:.*]] = "ttnn.add"(%[[LHS_SLICE]], %arg1)
  // CHECK-NOT: "ttnn.slice_static"(%[[ADD]])
  func.func @slice_eltwise_with_broadcast(%arg0: tensor<64x64xf32, #layout_64x64>, %arg1: tensor<1x1xf32, #layout_1x1>) -> tensor<32x32xf32, #layout_32x32> {
    %0 = "ttnn.add"(%arg0, %arg1) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<64x64xf32, #layout_64x64>, tensor<1x1xf32, #layout_1x1>) -> tensor<64x64xf32, #layout_64x64>
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
  // CHECK: %[[ADD:.*]] = "ttnn.add"(%[[R0]], %[[R1]])
  // CHECK-NOT: "ttnn.reshape"(%[[ADD]])
  func.func @reshape_eltwise(%arg0: tensor<1x1x1x1024xf32, #layout_r_1x1x1x1024>, %arg1: tensor<1x1x1x1024xf32, #layout_r_1x1x1x1024>) -> tensor<1x1x32x32xf32, #layout_r_1x1x32x32> {
    %0 = "ttnn.add"(%arg0, %arg1) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<1x1x1x1024xf32, #layout_r_1x1x1x1024>, tensor<1x1x1x1024xf32, #layout_r_1x1x1x1024>) -> tensor<1x1x1x1024xf32, #layout_r_1x1x1x1024>
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
  // permute-reshape row-major adjust:
  // CHECK-LABEL: func.func @permute_reshape_row_major_adjusting
  // CHECK: %[[RM_IN:.*]] = "ttnn.to_tensor_spec"(%arg0)
  // CHECK: %[[PERM:.*]] = "ttnn.permute"(%[[RM_IN]])
  // CHECK-SAME: permutation = array<i64: 1, 0>
  // CHECK-SAME: -> tensor<67108864x1xf32
  // CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"(%[[PERM]]) <{shape = [8192 : i32, 8192 : i32]}>
  // CHECK-SAME: -> tensor<8192x8192xf32
  // CHECK: %[[RESTORED:.*]] = "ttnn.to_tensor_spec"(%[[RESHAPE]])
  // CHECK: return %[[RESTORED]]
  func.func @permute_reshape_row_major_adjusting(%arg0: tensor<1x67108864xf32, #layout_1x67M_tile>) -> tensor<8192x8192xf32, #layout_8192x8192_tile> {
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 1, 0>}> : (tensor<1x67108864xf32, #layout_1x67M_tile>) -> tensor<67108864x1xf32, #layout_67Mx1_tile>
    %1 = "ttnn.reshape"(%0) <{shape = [8192 : i32, 8192 : i32]}> : (tensor<67108864x1xf32, #layout_67Mx1_tile>) -> tensor<8192x8192xf32, #layout_8192x8192_tile>
    return %1 : tensor<8192x8192xf32, #layout_8192x8192_tile>
  }

  // permute-repeat-reshape row-major adjust:
  // CHECK-LABEL: func.func @permute_repeat_reshape_row_major_adjusting
  // CHECK: %[[RM_IN:.*]] = "ttnn.to_tensor_spec"(%arg0)
  // CHECK: %[[PERM:.*]] = "ttnn.permute"(%[[RM_IN]])
  // CHECK-SAME: permutation = array<i64: 1, 0>
  // CHECK-SAME: -> tensor<67108864x1xf32
  // CHECK: %[[REPEAT:.*]] = "ttnn.repeat"(%[[PERM]]) <{repeat_dims = #ttnn.shape<1x2>}>
  // CHECK-SAME: -> tensor<67108864x2xf32
  // CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"(%[[REPEAT]]) <{shape = [8192 : i32, 16384 : i32]}>
  // CHECK-SAME: -> tensor<8192x16384xf32
  // CHECK: %[[RESTORED:.*]] = "ttnn.to_tensor_spec"(%[[RESHAPE]])
  // CHECK: return %[[RESTORED]]
  func.func @permute_repeat_reshape_row_major_adjusting(%arg0: tensor<1x67108864xf32, #layout_1x67M_tile>) -> tensor<8192x16384xf32, #layout_8192x16384_tile> {
    %0 = "ttnn.permute"(%arg0) <{permutation = array<i64: 1, 0>}> : (tensor<1x67108864xf32, #layout_1x67M_tile>) -> tensor<67108864x1xf32, #layout_67Mx1_tile>
    %1 = "ttnn.repeat"(%0) <{repeat_dims = #ttnn.shape<1x2>}> : (tensor<67108864x1xf32, #layout_67Mx1_tile>) -> tensor<67108864x2xf32, #layout_67Mx2_tile>
    %2 = "ttnn.reshape"(%1) <{shape = [8192 : i32, 16384 : i32]}> : (tensor<67108864x2xf32, #layout_67Mx2_tile>) -> tensor<8192x16384xf32, #layout_8192x16384_tile>
    return %2 : tensor<8192x16384xf32, #layout_8192x16384_tile>
  }

  // reshape-permute row-major adjust:
  // CHECK-LABEL: func.func @reshape_permute_row_major_adjusting
  // CHECK: %[[RM_IN:.*]] = "ttnn.to_tensor_spec"(%arg0)
  // CHECK: %[[RESHAPE:.*]] = "ttnn.reshape"(%[[RM_IN]]) <{shape = [8192 : i32, 8192 : i32, 1 : i32, 1 : i32]}>
  // CHECK-SAME: -> tensor<8192x8192x1x1xf32
  // CHECK: %[[PERM:.*]] = "ttnn.permute"(%[[RESHAPE]])
  // CHECK-SAME: permutation = array<i64: 2, 3, 0, 1>
  // CHECK-SAME: -> tensor<1x1x8192x8192xf32
  // CHECK: %[[RESTORED:.*]] = "ttnn.to_tensor_spec"(%[[PERM]])
  // CHECK: return %[[RESTORED]]
  func.func @reshape_permute_row_major_adjusting(%arg0: tensor<1x1x8192x8192xf32, #layout_4d_1x1x8192x8192_tile>) -> tensor<1x1x8192x8192xf32, #layout_4d_1x1x8192x8192_tile> {
    %0 = "ttnn.reshape"(%arg0) <{shape = [8192 : i32, 8192 : i32, 1 : i32, 1 : i32]}> : (tensor<1x1x8192x8192xf32, #layout_4d_1x1x8192x8192_tile>) -> tensor<8192x8192x1x1xf32, #layout_4d_8192x8192x1x1_tile>
    %1 = "ttnn.permute"(%0) <{permutation = array<i64: 2, 3, 0, 1>}> : (tensor<8192x8192x1x1xf32, #layout_4d_8192x8192x1x1_tile>) -> tensor<1x1x8192x8192xf32, #layout_4d_1x1x8192x8192_tile>
    return %1 : tensor<1x1x8192x8192xf32, #layout_4d_1x1x8192x8192_tile>
  }

  // a 1-D reshape whose only consumer is a to_tensor_spec(row_major) must emit its result directly in row-major
  // CHECK-LABEL: func.func @reshape_to_tensor_spec_row_major
  // CHECK: %[[RM_IN:.*]] = "ttnn.to_tensor_spec"(%arg0)
  // CHECK: "ttnn.reshape"(%[[RM_IN]])
  // CHECK-SAME: shape = [6935360 : i32]
  // CHECK-SAME: -> tensor<6935360xf32
  // CHECK-NOT: "ttnn.to_tensor_spec"
  // CHECK: return
  func.func @reshape_to_tensor_spec_row_major(%arg0: tensor<108365x64xf32, #layout_E_F>) -> tensor<6935360xf32, #layout_1d_rm> {
    %0 = "ttnn.reshape"(%arg0) <{shape = [6935360 : i32]}> : (tensor<108365x64xf32, #layout_E_F>) -> tensor<6935360xf32, #layout_1d_t>
    %1 = "ttnn.to_tensor_spec"(%0) : (tensor<6935360xf32, #layout_1d_t>) -> tensor<6935360xf32, #layout_1d_rm>
    return %1 : tensor<6935360xf32, #layout_1d_rm>
  }

  // a tile-aligned reshape result has equal tiled and row-major footprints, so the guard skips the rewrite
  // CHECK-LABEL: func.func @reshape_to_tensor_spec_tile_aligned
  // CHECK: %[[R:.*]] = "ttnn.reshape"
  // CHECK: "ttnn.to_tensor_spec"(%[[R]])
  // CHECK: return
  func.func @reshape_to_tensor_spec_tile_aligned(%arg0: tensor<128x32xf32, #layout_128x32>) -> tensor<64x64xf32, #layout_64x64_rm> {
    %0 = "ttnn.reshape"(%arg0) <{shape = [64 : i32, 64 : i32]}> : (tensor<128x32xf32, #layout_128x32>) -> tensor<64x64xf32, #layout_64x64>
    %1 = "ttnn.to_tensor_spec"(%0) : (tensor<64x64xf32, #layout_64x64>) -> tensor<64x64xf32, #layout_64x64_rm>
    return %1 : tensor<64x64xf32, #layout_64x64_rm>
  }

  // a consumer targeting row-major L1 rather than DRAM is resolved by the same single op
  // CHECK-LABEL: func.func @reshape_to_tensor_spec_l1_consumer
  // CHECK: %[[RM_IN:.*]] = "ttnn.to_tensor_spec"(%arg0)
  // CHECK: "ttnn.reshape"(%[[RM_IN]])
  // CHECK-SAME: -> tensor<6935360xf32
  // CHECK-NOT: "ttnn.to_tensor_spec"
  // CHECK: return
  func.func @reshape_to_tensor_spec_l1_consumer(%arg0: tensor<108365x64xf32, #layout_E_F>) -> tensor<6935360xf32, #layout_1d_rm_l1> {
    %0 = "ttnn.reshape"(%arg0) <{shape = [6935360 : i32]}> : (tensor<108365x64xf32, #layout_E_F>) -> tensor<6935360xf32, #layout_1d_t>
    %1 = "ttnn.to_tensor_spec"(%0) : (tensor<6935360xf32, #layout_1d_t>) -> tensor<6935360xf32, #layout_1d_rm_l1>
    return %1 : tensor<6935360xf32, #layout_1d_rm_l1>
  }

  // a second consumer needs the tiled result, so hasOneUse() blocks the rewrite
  // CHECK-LABEL: func.func @reshape_to_tensor_spec_multi_use
  // CHECK: %[[R:.*]] = "ttnn.reshape"(%arg0)
  // CHECK: "ttnn.to_tensor_spec"(%[[R]])
  func.func @reshape_to_tensor_spec_multi_use(%arg0: tensor<108365x64xf32, #layout_E_F>) -> (tensor<6935360xf32, #layout_1d_rm>, tensor<6935360xf32, #layout_1d_t>) {
    %0 = "ttnn.reshape"(%arg0) <{shape = [6935360 : i32]}> : (tensor<108365x64xf32, #layout_E_F>) -> tensor<6935360xf32, #layout_1d_t>
    %1 = "ttnn.to_tensor_spec"(%0) : (tensor<6935360xf32, #layout_1d_t>) -> tensor<6935360xf32, #layout_1d_rm>
    return %1, %0 : tensor<6935360xf32, #layout_1d_rm>, tensor<6935360xf32, #layout_1d_t>
  }

  // a to_tensor_spec producer can still yield TILE, so the input is gated on its layout; the inserted conversion then folds against it
  // CHECK-LABEL: func.func @reshape_to_tensor_spec_tiled_input
  // CHECK-NOT: "ttnn.to_tensor_spec"
  // CHECK: "ttnn.reshape"(%arg0)
  // CHECK-SAME: -> tensor<6935360xf32
  func.func @reshape_to_tensor_spec_tiled_input(%arg0: tensor<108365x64xf32, #layout_E_F_rm>) -> tensor<6935360xf32, #layout_1d_rm> {
    %0 = "ttnn.to_tensor_spec"(%arg0) : (tensor<108365x64xf32, #layout_E_F_rm>) -> tensor<108365x64xf32, #layout_E_F>
    %1 = "ttnn.reshape"(%0) <{shape = [6935360 : i32]}> : (tensor<108365x64xf32, #layout_E_F>) -> tensor<6935360xf32, #layout_1d_t>
    %2 = "ttnn.to_tensor_spec"(%1) : (tensor<6935360xf32, #layout_1d_t>) -> tensor<6935360xf32, #layout_1d_rm>
    return %2 : tensor<6935360xf32, #layout_1d_rm>
  }

  // the rewrite applies the same way when the flattened length is not a multiple of the tile height
  // CHECK-LABEL: func.func @reshape_to_tensor_spec_unaligned_flatten
  // CHECK: %[[RM_IN:.*]] = "ttnn.to_tensor_spec"(%arg0)
  // CHECK: "ttnn.reshape"(%[[RM_IN]])
  // CHECK-SAME: -> tensor<5000xf32
  // CHECK-NOT: "ttnn.to_tensor_spec"
  func.func @reshape_to_tensor_spec_unaligned_flatten(%arg0: tensor<100x50xf32, #layout_100x50>) -> tensor<5000xf32, #layout_5000_rm> {
    %0 = "ttnn.reshape"(%arg0) <{shape = [5000 : i32]}> : (tensor<100x50xf32, #layout_100x50>) -> tensor<5000xf32, #layout_5000_t>
    %1 = "ttnn.to_tensor_spec"(%0) : (tensor<5000xf32, #layout_5000_t>) -> tensor<5000xf32, #layout_5000_rm>
    return %1 : tensor<5000xf32, #layout_5000_rm>
  }
}
