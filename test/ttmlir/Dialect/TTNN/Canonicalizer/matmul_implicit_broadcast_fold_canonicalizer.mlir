// RUN: ttmlir-opt --canonicalize -o %t %s
// RUN: FileCheck --input-file=%t %s

// These tests are written in the TTNN dialect because the fold lives at the
// TTNN level: it is a `ttnn.matmul` canonicalization that drops a `ttnn.repeat`
// feeding its RHS when TTNN's matmul implicitly reproduces that broadcast.
// `ttir.broadcast` lowers to `ttnn.repeat`, so the cases below mirror the TTIR
// shapes after lowering.

#dram = #ttnn.buffer_type<dram>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2, d3), <1x1>, memref<1024x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<4x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 544 + d1, d2), <1x1>, memref<17x90x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 2880 + d1, d2), <1x1>, memref<90x180x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 544 + d1, d2), <1x1>, memref<68x180x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 2880 + d1, d2), <1x1>, memref<360x180x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout6 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 32768 + d1 * 128 + d2, d3), <1x1>, memref<4096x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout7 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 128 + d2, d3), <1x1>, memref<128x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout8 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 128 + d1 * 128 + d2, d3), <1x1>, memref<16x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
module {
  // The supported case: a rank-4 RHS whose batch dim 1 is repeated to match the
  // LHS. TTNN's matmul broadcasts dim 1 implicitly, so the explicit
  // ttnn.repeat is dropped and the matmul consumes the unrepeated operand.
  // CHECK-LABEL: func.func @fold_rhs_batch_dim1
  func.func @fold_rhs_batch_dim1(%arg0: tensor<1x256x128x128xf32, #ttnn_layout>, %arg1: tensor<1x1x128x128xf32, #ttnn_layout1>) -> tensor<1x256x128x128xf32, #ttnn_layout> {
    %0 = "ttnn.repeat"(%arg1) <{repeat_dims = #ttnn.shape<1x256x1x1>}> : (tensor<1x1x128x128xf32, #ttnn_layout1>) -> tensor<1x256x128x128xf32, #ttnn_layout>
    %1 = "ttnn.matmul"(%arg0, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x256x128x128xf32, #ttnn_layout>, tensor<1x256x128x128xf32, #ttnn_layout>) -> tensor<1x256x128x128xf32, #ttnn_layout>
    // CHECK-NOT: "ttnn.repeat"
    // CHECK: "ttnn.matmul"(%arg0, %arg1)
    // CHECK-SAME: (tensor<1x256x128x128xf32, #ttnn_layout>, tensor<1x1x128x128xf32, #ttnn_layout1>) -> tensor<1x256x128x128xf32, #ttnn_layout>
    return %1 : tensor<1x256x128x128xf32, #ttnn_layout>
  }

  // Negative: a repeat on the LHS is out of scope -- the fold only targets the
  // RHS -- so it must be kept.
  // CHECK-LABEL: func.func @no_fold_lhs_batch_dim1
  func.func @no_fold_lhs_batch_dim1(%arg0: tensor<1x1x128x128xf32, #ttnn_layout1>, %arg1: tensor<1x256x128x128xf32, #ttnn_layout>) -> tensor<1x256x128x128xf32, #ttnn_layout> {
    %0 = "ttnn.repeat"(%arg0) <{repeat_dims = #ttnn.shape<1x256x1x1>}> : (tensor<1x1x128x128xf32, #ttnn_layout1>) -> tensor<1x256x128x128xf32, #ttnn_layout>
    %1 = "ttnn.matmul"(%0, %arg1) <{transpose_a = false, transpose_b = false}> : (tensor<1x256x128x128xf32, #ttnn_layout>, tensor<1x256x128x128xf32, #ttnn_layout>) -> tensor<1x256x128x128xf32, #ttnn_layout>
    // CHECK: "ttnn.repeat"
    // CHECK: "ttnn.matmul"
    return %1 : tensor<1x256x128x128xf32, #ttnn_layout>
  }

  // Negative: rank-3 repeat on dim 0. TTNN's matmul cannot broadcast dim 0, so
  // the repeat must be kept (this is the shape that previously faulted).
  // CHECK-LABEL: func.func @no_fold_rank3_batch_dim0
  func.func @no_fold_rank3_batch_dim0(%arg0: tensor<1x544x2880xf32, #ttnn_layout2>, %arg1: tensor<1x2880x5760xf32, #ttnn_layout3>) -> tensor<4x544x5760xf32, #ttnn_layout4> {
    %0 = "ttnn.repeat"(%arg1) <{repeat_dims = #ttnn.shape<4x1x1>}> : (tensor<1x2880x5760xf32, #ttnn_layout3>) -> tensor<4x2880x5760xf32, #ttnn_layout5>
    %1 = "ttnn.matmul"(%arg0, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x544x2880xf32, #ttnn_layout2>, tensor<4x2880x5760xf32, #ttnn_layout5>) -> tensor<4x544x5760xf32, #ttnn_layout4>
    // CHECK: "ttnn.repeat"
    // CHECK: "ttnn.matmul"
    return %1 : tensor<4x544x5760xf32, #ttnn_layout4>
  }

  // Negative: a rank-4 repeat on dim 0. The fold only targets dim 1, so a
  // repeat that touches any other dim must be kept.
  // CHECK-LABEL: func.func @no_fold_rank4_batch_dim0
  func.func @no_fold_rank4_batch_dim0(%arg0: tensor<4x256x128x128xf32, #ttnn_layout6>, %arg1: tensor<1x256x128x128xf32, #ttnn_layout>) -> tensor<4x256x128x128xf32, #ttnn_layout6> {
    %0 = "ttnn.repeat"(%arg1) <{repeat_dims = #ttnn.shape<4x1x1x1>}> : (tensor<1x256x128x128xf32, #ttnn_layout>) -> tensor<4x256x128x128xf32, #ttnn_layout6>
    %1 = "ttnn.matmul"(%arg0, %0) <{transpose_a = false, transpose_b = false}> : (tensor<4x256x128x128xf32, #ttnn_layout6>, tensor<4x256x128x128xf32, #ttnn_layout6>) -> tensor<4x256x128x128xf32, #ttnn_layout6>
    // CHECK: "ttnn.repeat"
    // CHECK: "ttnn.matmul"
    return %1 : tensor<4x256x128x128xf32, #ttnn_layout6>
  }

  // Negative: the repeat is on dim 1 of the RHS, but the LHS dim 1 is itself
  // size 1 -- so the matmul's dim-1 output comes entirely from the RHS repeat.
  // Dropping it would shrink the result from 256 to 1, so it must be kept. This
  // is why "repeat on dim 1 of the RHS" alone is not sufficient.
  // CHECK-LABEL: func.func @no_fold_lhs_dim1_is_one
  func.func @no_fold_lhs_dim1_is_one(%arg0: tensor<1x1x128x128xf32, #ttnn_layout1>, %arg1: tensor<1x1x128x128xf32, #ttnn_layout1>) -> tensor<1x256x128x128xf32, #ttnn_layout> {
    %0 = "ttnn.repeat"(%arg1) <{repeat_dims = #ttnn.shape<1x256x1x1>}> : (tensor<1x1x128x128xf32, #ttnn_layout1>) -> tensor<1x256x128x128xf32, #ttnn_layout>
    %1 = "ttnn.matmul"(%arg0, %0) <{transpose_a = false, transpose_b = false}> : (tensor<1x1x128x128xf32, #ttnn_layout1>, tensor<1x256x128x128xf32, #ttnn_layout>) -> tensor<1x256x128x128xf32, #ttnn_layout>
    // CHECK: "ttnn.repeat"
    // CHECK: "ttnn.matmul"
    return %1 : tensor<1x256x128x128xf32, #ttnn_layout>
  }

  // Negative: dim 1 is repeated and the LHS supplies its size, but the outer
  // batch dim 0 is > 1. After folding the RHS would be 4x1x128x128, whose batch
  // size (4) is not 1, so metal would NOT implicitly broadcast the RHS batch and
  // would instead fault with a batch dimension mismatch. The repeat must be
  // kept. This is the real-world shape the tightening protects.
  // CHECK-LABEL: func.func @no_fold_outer_batch_dim0_gt_one
  func.func @no_fold_outer_batch_dim0_gt_one(%arg0: tensor<4x8x128x128xf32, #ttnn_layout7>, %arg1: tensor<4x1x128x128xf32, #ttnn_layout8>) -> tensor<4x8x128x128xf32, #ttnn_layout7> {
    %0 = "ttnn.repeat"(%arg1) <{repeat_dims = #ttnn.shape<1x8x1x1>}> : (tensor<4x1x128x128xf32, #ttnn_layout8>) -> tensor<4x8x128x128xf32, #ttnn_layout7>
    %1 = "ttnn.matmul"(%arg0, %0) <{transpose_a = false, transpose_b = false}> : (tensor<4x8x128x128xf32, #ttnn_layout7>, tensor<4x8x128x128xf32, #ttnn_layout7>) -> tensor<4x8x128x128xf32, #ttnn_layout7>
    // CHECK: "ttnn.repeat"
    // CHECK: "ttnn.matmul"
    return %1 : tensor<4x8x128x128xf32, #ttnn_layout7>
  }
}
