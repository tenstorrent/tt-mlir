// Pure D2MGenericFusion isolation test. Input is a hand-crafted post-LTL
// snapshot: every `d2m.to_layout` has already been materialized as its own
// `d2m.generic`. The test exercises ONLY the fusion pass plus
// `--canonicalize` to drop the dead tilize-producer.
//
// Three cases:
//   * `@tilize_into_matmul`: tilize-`d2m.generic` (scalar→tile) feeds matmul
//     consumer whose interface op (`tile_matmul`) declares
//     `canConsumeUntilizedOperand(0)`. Expect fusion: matmul generic's
//     outer A becomes scalar, body gets an inlined `d2m.tile_tilize_block`.
//   * `@untilize_from_matmul`: matmul-`d2m.generic` (tile→tile) feeds an
//     untilize consumer (tile→scalar). With `enable-l1-acc=false`, matmul's
//     `canProduceUntilizedResult(0, /*l1Acc=*/false)` is true → fusion folds
//     the untilize into the matmul producer's outer result.
//   * `@already_tiled`: no tilize-producer in front of matmul. Negative
//     check that the pattern does not fire when there's nothing to fold.

// RUN: ttmlir-opt --ttcore-register-device \
// RUN:   --d2m-generic-fusion="enable-l1-acc=false" --canonicalize \
// RUN:   -o %t %s
// RUN: FileCheck %s --input-file=%t

#layout    = #ttcore.metal_layout<logical_shape = 128x96, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layout1   = #ttcore.metal_layout<logical_shape = 128x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layout2   = #ttcore.metal_layout<logical_shape = 96x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>

#mapp  = affine_map<(d0, d1) -> (d0, d1)>
#mapA  = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapB  = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapC  = affine_map<(d0, d1, d2) -> (d0, d1)>

#parallel  = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

// CHECK-LABEL: func.func @tilize_into_matmul
// After fusion: only ONE `d2m.generic` remains, and it carries the
// row-major operand on its outer ins() list with an inlined
// `tile_tilize_block` in the body. The negative ensures the upstream
// tilize generic is gone — only the fused matmul generic is left.
// CHECK: d2m.generic
// CHECK: ins(%arg0
// CHECK-SAME: tensor<1x1x128x96xf32, #layout>
// CHECK: d2m.tile_tilize_block
// CHECK: linalg.generic
// CHECK: d2m.tile_matmul
// CHECK-NOT: "d2m.tile_tilize_block"
// CHECK-NOT: d2m.generic
func.func @tilize_into_matmul(%rm_a: tensor<1x1x128x96xf32, #layout>) -> tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1> {
  %tiled_empty = d2m.empty() : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>
  %tilized = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapp, #mapp], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%rm_a : tensor<1x1x128x96xf32, #layout>)
      outs(%tiled_empty : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>) {
    %b0 = d2m.block_index(0) : index
    %b1 = d2m.block_index(1) : index
    %rm_buf = tensor.empty() : tensor<128x96xf32>
    %rm = d2m.remote_load %rm_buf %rm_a[%b0, %b1] : tensor<128x96xf32>, tensor<1x1x128x96xf32, #layout> -> tensor<128x96xf32>
    %tile_buf = tensor.empty() : tensor<4x3x!ttcore.tile<32x32, f32>>
    %tile = "d2m.tile_tilize_block"(%rm, %tile_buf) : (tensor<128x96xf32>, tensor<4x3x!ttcore.tile<32x32, f32>>) -> tensor<4x3x!ttcore.tile<32x32, f32>>
    %s = d2m.remote_store %tiled_empty[%b0, %b1] %tile : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>, tensor<4x3x!ttcore.tile<32x32, f32>> -> tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>
    d2m.yield %s : (tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>)
  } : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>

  %b_empty   = d2m.empty() : tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout2>
  %out_empty = d2m.empty() : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
  %r = d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapA, #mapB, #mapC], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
      ins(%tilized, %b_empty : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout2>)
      outs(%out_empty : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>) {
    %b0 = d2m.block_index(0) : index
    %b1 = d2m.block_index(1) : index
    %b2 = d2m.block_index(2) : index
    %buf_a = tensor.empty() : tensor<4x3x!ttcore.tile<32x32, f32>>
    %la = d2m.remote_load %buf_a %tilized[%b0, %b2] : tensor<4x3x!ttcore.tile<32x32, f32>>, tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout> -> tensor<4x3x!ttcore.tile<32x32, f32>>
    %buf_b = tensor.empty() : tensor<3x2x!ttcore.tile<32x32, f32>>
    %lb = d2m.remote_load %buf_b %b_empty[%b2, %b1] : tensor<3x2x!ttcore.tile<32x32, f32>>, tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout2> -> tensor<3x2x!ttcore.tile<32x32, f32>>
    %buf_out = tensor.empty() : tensor<4x2x!ttcore.tile<32x32, f32>>
    %mm = linalg.generic {indexing_maps = [#mapA, #mapB, #mapC], iterator_types = ["parallel", "parallel", "reduction"]} ins(%la, %lb : tensor<4x3x!ttcore.tile<32x32, f32>>, tensor<3x2x!ttcore.tile<32x32, f32>>) outs(%buf_out : tensor<4x2x!ttcore.tile<32x32, f32>>) {
    ^bb0(%in: !ttcore.tile<32x32, f32>, %in_b: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
      %p = "d2m.tile_matmul"(%in, %in_b, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %p : !ttcore.tile<32x32, f32>
    } -> tensor<4x2x!ttcore.tile<32x32, f32>>
    %bo0 = d2m.block_index(0) : index
    %bo1 = d2m.block_index(1) : index
    %s = d2m.remote_store %out_empty[%bo0, %bo1] %mm : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>, tensor<4x2x!ttcore.tile<32x32, f32>> -> tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
    d2m.yield %s : (tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>)
  } : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
  return %r : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
}

// CHECK-LABEL: func.func @untilize_from_matmul
// After fusion: only ONE `d2m.generic` remains, its outer C result is the
// row-major destination, and `tile_untilize_block` is inlined in the body
// between the matmul and the remote_store.
// CHECK: d2m.generic
// CHECK: outs(%{{[a-z0-9_]+}}
// CHECK-SAME: tensor<1x1x128x64xf32, #layout
// CHECK: d2m.tile_matmul
// CHECK: d2m.tile_untilize_block
// CHECK: d2m.remote_store
// CHECK-NOT: d2m.generic
func.func @untilize_from_matmul(%tile_a: tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>,
                                %tile_b: tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout2>) -> tensor<1x1x128x64xf32, #layout1> {
  %tile_out_empty = d2m.empty() : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
  %tile_out = d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapA, #mapB, #mapC], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
      ins(%tile_a, %tile_b : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout2>)
      outs(%tile_out_empty : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>) {
    %b0 = d2m.block_index(0) : index
    %b1 = d2m.block_index(1) : index
    %b2 = d2m.block_index(2) : index
    %buf_a = tensor.empty() : tensor<4x3x!ttcore.tile<32x32, f32>>
    %la = d2m.remote_load %buf_a %tile_a[%b0, %b2] : tensor<4x3x!ttcore.tile<32x32, f32>>, tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout> -> tensor<4x3x!ttcore.tile<32x32, f32>>
    %buf_b = tensor.empty() : tensor<3x2x!ttcore.tile<32x32, f32>>
    %lb = d2m.remote_load %buf_b %tile_b[%b2, %b1] : tensor<3x2x!ttcore.tile<32x32, f32>>, tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout2> -> tensor<3x2x!ttcore.tile<32x32, f32>>
    %buf_out = tensor.empty() : tensor<4x2x!ttcore.tile<32x32, f32>>
    %mm = linalg.generic {indexing_maps = [#mapA, #mapB, #mapC], iterator_types = ["parallel", "parallel", "reduction"]} ins(%la, %lb : tensor<4x3x!ttcore.tile<32x32, f32>>, tensor<3x2x!ttcore.tile<32x32, f32>>) outs(%buf_out : tensor<4x2x!ttcore.tile<32x32, f32>>) {
    ^bb0(%in: !ttcore.tile<32x32, f32>, %in_b: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
      %p = "d2m.tile_matmul"(%in, %in_b, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %p : !ttcore.tile<32x32, f32>
    } -> tensor<4x2x!ttcore.tile<32x32, f32>>
    %bo0 = d2m.block_index(0) : index
    %bo1 = d2m.block_index(1) : index
    %s = d2m.remote_store %tile_out_empty[%bo0, %bo1] %mm : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>, tensor<4x2x!ttcore.tile<32x32, f32>> -> tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
    d2m.yield %s : (tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>)
  } : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
  %rm_out_empty = d2m.empty() : tensor<1x1x128x64xf32, #layout1>
  %rm_out = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapp, #mapp], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%tile_out : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>)
      outs(%rm_out_empty : tensor<1x1x128x64xf32, #layout1>) {
    %b0 = d2m.block_index(0) : index
    %b1 = d2m.block_index(1) : index
    %tile_buf = tensor.empty() : tensor<4x2x!ttcore.tile<32x32, f32>>
    %tile = d2m.remote_load %tile_buf %tile_out[%b0, %b1] : tensor<4x2x!ttcore.tile<32x32, f32>>, tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1> -> tensor<4x2x!ttcore.tile<32x32, f32>>
    %rm_buf = tensor.empty() : tensor<128x64xf32>
    %rm = "d2m.tile_untilize_block"(%tile, %rm_buf) : (tensor<4x2x!ttcore.tile<32x32, f32>>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %s = d2m.remote_store %rm_out_empty[%b0, %b1] %rm : tensor<1x1x128x64xf32, #layout1>, tensor<128x64xf32> -> tensor<1x1x128x64xf32, #layout1>
    d2m.yield %s : (tensor<1x1x128x64xf32, #layout1>)
  } : tensor<1x1x128x64xf32, #layout1>
  return %rm_out : tensor<1x1x128x64xf32, #layout1>
}

// CHECK-LABEL: func.func @rm_in_rm_out_matmul
// Both directions on one matmul: tilize-producer feeds srcA AND
// untilize-consumer reads result. Greedy rewriter applies both folds in
// some order; after fixed-point ONE generic remains, RM operand in, RM
// result out, with tile_tilize_block AND tile_untilize_block inlined in
// its body.
// CHECK: d2m.generic
// CHECK: ins(%arg0
// CHECK-SAME: tensor<1x1x128x96xf32, #layout>
// CHECK: d2m.tile_tilize_block
// CHECK: d2m.tile_matmul
// CHECK: d2m.tile_untilize_block
// CHECK: d2m.remote_store
// CHECK-NOT: d2m.generic
func.func @rm_in_rm_out_matmul(%rm_a: tensor<1x1x128x96xf32, #layout>,
                                %tile_b: tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout2>) -> tensor<1x1x128x64xf32, #layout1> {
  %tiled_empty = d2m.empty() : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>
  %tilized = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapp, #mapp], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%rm_a : tensor<1x1x128x96xf32, #layout>)
      outs(%tiled_empty : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>) {
    %b0 = d2m.block_index(0) : index
    %b1 = d2m.block_index(1) : index
    %rm_buf = tensor.empty() : tensor<128x96xf32>
    %rm = d2m.remote_load %rm_buf %rm_a[%b0, %b1] : tensor<128x96xf32>, tensor<1x1x128x96xf32, #layout> -> tensor<128x96xf32>
    %tile_buf = tensor.empty() : tensor<4x3x!ttcore.tile<32x32, f32>>
    %tile = "d2m.tile_tilize_block"(%rm, %tile_buf) : (tensor<128x96xf32>, tensor<4x3x!ttcore.tile<32x32, f32>>) -> tensor<4x3x!ttcore.tile<32x32, f32>>
    %s = d2m.remote_store %tiled_empty[%b0, %b1] %tile : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>, tensor<4x3x!ttcore.tile<32x32, f32>> -> tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>
    d2m.yield %s : (tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>)
  } : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>

  %tile_out_empty = d2m.empty() : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
  %tile_out = d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapA, #mapB, #mapC], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
      ins(%tilized, %tile_b : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout2>)
      outs(%tile_out_empty : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>) {
    %b0 = d2m.block_index(0) : index
    %b1 = d2m.block_index(1) : index
    %b2 = d2m.block_index(2) : index
    %buf_a = tensor.empty() : tensor<4x3x!ttcore.tile<32x32, f32>>
    %la = d2m.remote_load %buf_a %tilized[%b0, %b2] : tensor<4x3x!ttcore.tile<32x32, f32>>, tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout> -> tensor<4x3x!ttcore.tile<32x32, f32>>
    %buf_b = tensor.empty() : tensor<3x2x!ttcore.tile<32x32, f32>>
    %lb = d2m.remote_load %buf_b %tile_b[%b2, %b1] : tensor<3x2x!ttcore.tile<32x32, f32>>, tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout2> -> tensor<3x2x!ttcore.tile<32x32, f32>>
    %buf_out = tensor.empty() : tensor<4x2x!ttcore.tile<32x32, f32>>
    %mm = linalg.generic {indexing_maps = [#mapA, #mapB, #mapC], iterator_types = ["parallel", "parallel", "reduction"]} ins(%la, %lb : tensor<4x3x!ttcore.tile<32x32, f32>>, tensor<3x2x!ttcore.tile<32x32, f32>>) outs(%buf_out : tensor<4x2x!ttcore.tile<32x32, f32>>) {
    ^bb0(%in: !ttcore.tile<32x32, f32>, %in_b: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
      %p = "d2m.tile_matmul"(%in, %in_b, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %p : !ttcore.tile<32x32, f32>
    } -> tensor<4x2x!ttcore.tile<32x32, f32>>
    %bo0 = d2m.block_index(0) : index
    %bo1 = d2m.block_index(1) : index
    %s = d2m.remote_store %tile_out_empty[%bo0, %bo1] %mm : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>, tensor<4x2x!ttcore.tile<32x32, f32>> -> tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
    d2m.yield %s : (tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>)
  } : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>

  %rm_out_empty = d2m.empty() : tensor<1x1x128x64xf32, #layout1>
  %rm_out = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapp, #mapp], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
      ins(%tile_out : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>)
      outs(%rm_out_empty : tensor<1x1x128x64xf32, #layout1>) {
    %b0 = d2m.block_index(0) : index
    %b1 = d2m.block_index(1) : index
    %tile_buf = tensor.empty() : tensor<4x2x!ttcore.tile<32x32, f32>>
    %tile = d2m.remote_load %tile_buf %tile_out[%b0, %b1] : tensor<4x2x!ttcore.tile<32x32, f32>>, tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1> -> tensor<4x2x!ttcore.tile<32x32, f32>>
    %rm_buf = tensor.empty() : tensor<128x64xf32>
    %rm = "d2m.tile_untilize_block"(%tile, %rm_buf) : (tensor<4x2x!ttcore.tile<32x32, f32>>, tensor<128x64xf32>) -> tensor<128x64xf32>
    %s = d2m.remote_store %rm_out_empty[%b0, %b1] %rm : tensor<1x1x128x64xf32, #layout1>, tensor<128x64xf32> -> tensor<1x1x128x64xf32, #layout1>
    d2m.yield %s : (tensor<1x1x128x64xf32, #layout1>)
  } : tensor<1x1x128x64xf32, #layout1>
  return %rm_out : tensor<1x1x128x64xf32, #layout1>
}

// CHECK-LABEL: func.func @already_tiled
// Negative: no tilize-producer to absorb → matmul body must NOT contain
// `tile_tilize_block`.
// CHECK-NOT: d2m.tile_tilize_block
// CHECK: d2m.generic
// CHECK: d2m.tile_matmul
func.func @already_tiled(%tile_a: tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1> {
  %b_empty   = d2m.empty() : tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout2>
  %out_empty = d2m.empty() : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
  %r = d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapA, #mapB, #mapC], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
      ins(%tile_a, %b_empty : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout2>)
      outs(%out_empty : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>) {
    %b0 = d2m.block_index(0) : index
    %b1 = d2m.block_index(1) : index
    %b2 = d2m.block_index(2) : index
    %buf_a = tensor.empty() : tensor<4x3x!ttcore.tile<32x32, f32>>
    %la = d2m.remote_load %buf_a %tile_a[%b0, %b2] : tensor<4x3x!ttcore.tile<32x32, f32>>, tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout> -> tensor<4x3x!ttcore.tile<32x32, f32>>
    %buf_b = tensor.empty() : tensor<3x2x!ttcore.tile<32x32, f32>>
    %lb = d2m.remote_load %buf_b %b_empty[%b2, %b1] : tensor<3x2x!ttcore.tile<32x32, f32>>, tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout2> -> tensor<3x2x!ttcore.tile<32x32, f32>>
    %buf_out = tensor.empty() : tensor<4x2x!ttcore.tile<32x32, f32>>
    %mm = linalg.generic {indexing_maps = [#mapA, #mapB, #mapC], iterator_types = ["parallel", "parallel", "reduction"]} ins(%la, %lb : tensor<4x3x!ttcore.tile<32x32, f32>>, tensor<3x2x!ttcore.tile<32x32, f32>>) outs(%buf_out : tensor<4x2x!ttcore.tile<32x32, f32>>) {
    ^bb0(%in: !ttcore.tile<32x32, f32>, %in_b: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
      %p = "d2m.tile_matmul"(%in, %in_b, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %p : !ttcore.tile<32x32, f32>
    } -> tensor<4x2x!ttcore.tile<32x32, f32>>
    %bo0 = d2m.block_index(0) : index
    %bo1 = d2m.block_index(1) : index
    %s = d2m.remote_store %out_empty[%bo0, %bo1] %mm : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>, tensor<4x2x!ttcore.tile<32x32, f32>> -> tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
    d2m.yield %s : (tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>)
  } : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
  return %r : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout1>
}
