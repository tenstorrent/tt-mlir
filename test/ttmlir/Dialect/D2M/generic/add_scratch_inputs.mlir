// RUN: ttmlir-opt --ttcore-register-device --d2m-add-scratch-inputs %s | FileCheck %s

#layout = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

!tile_f32 = !ttcore.tile<32x32, f32>
!tiled = tensor<1x1x4x4x!tile_f32, #layout>
!shard = tensor<4x4x!tile_f32>
!cb = !d2m.cb<!shard>

#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// Two tile_add ops in a fused generic → scratch should be added at input index 2.
// Verify: scratch empty (1x1x1x32 for f32 @ 128KB), scratch_inputs attr, and
// the scratch tensor appears as the 3rd input to the generic.

// CHECK-LABEL: func.func @two_adds_gets_scratch
// CHECK: d2m.empty() : tensor<1x1x1x32x!ttcore.tile<32x32, f32>
// CHECK: d2m.generic
// CHECK-SAME: scratch_inputs = array<i64: 2>
// CHECK: ins(%{{.*}}, %{{.*}}, %{{.*}} :
// CHECK-SAME: tensor<1x1x1x32x!ttcore.tile<32x32, f32>
func.func @two_adds_gets_scratch(%arg0: !tiled, %arg1: !tiled) -> !tiled {
  %out = d2m.empty() : !tiled
  %result = d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [#map, #map, #map],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%arg0, %arg1 : !tiled, !tiled)
  outs(%out : !tiled) {
  ^unified0(%cb0: !cb, %cb1: !cb, %cb2: !cb):
    %block0 = d2m.block_index(0) : index
    %block1 = d2m.block_index(1) : index
    %e0 = tensor.empty() : !shard
    %a = d2m.remote_load %e0 %arg0[%block0, %block1] : !shard, !tiled -> !shard
    %e1 = tensor.empty() : !shard
    %b = d2m.remote_load %e1 %arg1[%block0, %block1] : !shard, !tiled -> !shard
    // First add: tmp = a + b.
    %e2 = tensor.empty() : !shard
    %add1 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : !shard, !shard) outs(%e2 : !shard) {
    ^bb0(%in0: !tile_f32, %in1: !tile_f32, %unused: !tile_f32):
      %s = "d2m.tile_add"(%in0, %in1) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %s : !tile_f32
    } -> !shard
    // Second add: result = tmp + a.
    %e3 = tensor.empty() : !shard
    %add2 = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%add1, %a : !shard, !shard) outs(%e3 : !shard) {
    ^bb0(%in0: !tile_f32, %in1: !tile_f32, %unused: !tile_f32):
      %s = "d2m.tile_add"(%in0, %in1) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %s : !tile_f32
    } -> !shard
    %stored = d2m.remote_store %out[%block0, %block1] %add2 : !tiled, !shard -> !tiled
    d2m.yield %stored : (!tiled)
  } : !tiled
  return %result : !tiled
}

// Mixed binary FPU ops (tile_add + tile_mul) → scratch should be added.

// CHECK-LABEL: func.func @add_and_mul_gets_scratch
// CHECK: d2m.empty() : tensor<1x1x1x32x!ttcore.tile<32x32, f32>
// CHECK: d2m.generic
// CHECK-SAME: scratch_inputs = array<i64: 2>
// CHECK: ins(%{{.*}}, %{{.*}}, %{{.*}} :
// CHECK-SAME: tensor<1x1x1x32x!ttcore.tile<32x32, f32>
func.func @add_and_mul_gets_scratch(%arg0: !tiled, %arg1: !tiled) -> !tiled {
  %out = d2m.empty() : !tiled
  %result = d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [#map, #map, #map],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%arg0, %arg1 : !tiled, !tiled)
  outs(%out : !tiled) {
  ^unified0(%cb0: !cb, %cb1: !cb, %cb2: !cb):
    %block0 = d2m.block_index(0) : index
    %block1 = d2m.block_index(1) : index
    %e0 = tensor.empty() : !shard
    %a = d2m.remote_load %e0 %arg0[%block0, %block1] : !shard, !tiled -> !shard
    %e1 = tensor.empty() : !shard
    %b = d2m.remote_load %e1 %arg1[%block0, %block1] : !shard, !tiled -> !shard
    // Add: tmp = a + b.
    %e2 = tensor.empty() : !shard
    %add = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : !shard, !shard) outs(%e2 : !shard) {
    ^bb0(%in0: !tile_f32, %in1: !tile_f32, %unused: !tile_f32):
      %s = "d2m.tile_add"(%in0, %in1) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %s : !tile_f32
    } -> !shard
    // Mul: result = tmp * a.
    %e3 = tensor.empty() : !shard
    %mul = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%add, %a : !shard, !shard) outs(%e3 : !shard) {
    ^bb0(%in0: !tile_f32, %in1: !tile_f32, %unused: !tile_f32):
      %s = "d2m.tile_mul"(%in0, %in1) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %s : !tile_f32
    } -> !shard
    %stored = d2m.remote_store %out[%block0, %block1] %mul : !tiled, !shard -> !tiled
    d2m.yield %stored : (!tiled)
  } : !tiled
  return %result : !tiled
}

// Single tile_add → no scratch (needsScratch requires > 1 binary FPU op).

// CHECK-LABEL: func.func @single_add_no_scratch
// CHECK-NOT: scratch_inputs
// CHECK: return
func.func @single_add_no_scratch(%arg0: !tiled, %arg1: !tiled) -> !tiled {
  %out = d2m.empty() : !tiled
  %result = d2m.generic {
    block_factors = [1, 1], grid = #ttcore.grid<1x1>,
    indexing_maps = [#map, #map, #map],
    iterator_types = [#parallel, #parallel],
    threads = [#d2m.thread<unified>]
  }
  ins(%arg0, %arg1 : !tiled, !tiled)
  outs(%out : !tiled) {
  ^unified0(%cb0: !cb, %cb1: !cb, %cb2: !cb):
    %block0 = d2m.block_index(0) : index
    %block1 = d2m.block_index(1) : index
    %e0 = tensor.empty() : !shard
    %a = d2m.remote_load %e0 %arg0[%block0, %block1] : !shard, !tiled -> !shard
    %e1 = tensor.empty() : !shard
    %b = d2m.remote_load %e1 %arg1[%block0, %block1] : !shard, !tiled -> !shard
    %e2 = tensor.empty() : !shard
    %add = linalg.generic {
      indexing_maps = [#map, #map, #map],
      iterator_types = ["parallel", "parallel"]
    } ins(%a, %b : !shard, !shard) outs(%e2 : !shard) {
    ^bb0(%in0: !tile_f32, %in1: !tile_f32, %unused: !tile_f32):
      %s = "d2m.tile_add"(%in0, %in1) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %s : !tile_f32
    } -> !shard
    %stored = d2m.remote_store %out[%block0, %block1] %add : !tiled, !shard -> !tiled
    d2m.yield %stored : (!tiled)
  } : !tiled
  return %result : !tiled
}
