// RUN: ttmlir-opt --d2m-optimize-masks %s | FileCheck %s

#layout_unaligned = #ttcore.metal_layout<logical_shape = 50x50, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layout_aligned = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

!tile_f32 = !ttcore.tile<32x32, f32>

// CHECK-LABEL: func.func @remove_back_to_back_same_fill
func.func @remove_back_to_back_same_fill(
  %arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>,
  %arg1: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>,
  %arg2: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>
) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> {
  // CHECK: %[[MASK:.*]] = d2m.mask
  // CHECK-NOT: d2m.mask
  // CHECK: return %[[MASK]]
  %0 = d2m.mask %arg0, %arg1 logical_shape = [50, 50] fill_value = <zero> : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> into tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>
  %1 = d2m.mask %0, %arg2 logical_shape = [50, 50] fill_value = <zero> : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> into tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>
  return %1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>
}

// CHECK-LABEL: func.func @keep_changed_fill
func.func @keep_changed_fill(
  %arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>,
  %arg1: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>,
  %arg2: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>
) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> {
  // CHECK: d2m.mask
  // CHECK: d2m.mask
  // CHECK: return
  %0 = d2m.mask %arg0, %arg1 logical_shape = [50, 50] fill_value = <zero> : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> into tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>
  %1 = d2m.mask %0, %arg2 logical_shape = [50, 50] fill_value = <one> : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> into tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>
  return %1 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_unaligned>
}

// CHECK-LABEL: func.func @remove_aligned_mask
func.func @remove_aligned_mask(
  %arg0: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned>,
  %arg1: tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned>
) -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned> {
  // CHECK-NOT: d2m.mask
  // CHECK: return %arg0
  %0 = d2m.mask %arg0, %arg1 logical_shape = [64, 64] fill_value = <zero> : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned> into tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned> -> tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned>
  return %0 : tensor<1x1x2x2x!ttcore.tile<32x32, f32>, #layout_aligned>
}

// CHECK-LABEL: func.func @remove_generic_add_zero_one_result_mask
func.func @remove_generic_add_zero_one_result_mask(
  %arg0: tensor<4x4x!tile_f32>,
  %arg1: tensor<4x4x!tile_f32>
) -> tensor<4x4x!tile_f32> {
  %out0 = d2m.empty() : tensor<4x4x!tile_f32>
  %out1 = d2m.empty() : tensor<4x4x!tile_f32>
  %out2 = d2m.empty() : tensor<4x4x!tile_f32>
  %out3 = d2m.empty() : tensor<4x4x!tile_f32>
  // CHECK: d2m.mask {{.*}} fill_value = <zero>
  // CHECK: d2m.mask {{.*}} fill_value = <one>
  %0 = d2m.mask %arg0, %out0 logical_shape = [50, 50] fill_value = <zero> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  %1 = d2m.mask %arg1, %out1 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  // CHECK: d2m.generic
  // CHECK-NOT: d2m.mask
  // CHECK: return
  %2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]}
      ins(%0, %1 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>)
      outs(%out2 : tensor<4x4x!tile_f32>) {
  ^unified0:
    %result = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%0, %1 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>) outs(%out2 : tensor<4x4x!tile_f32>) {
    ^bb0(%lhs: !tile_f32, %rhs: !tile_f32, %out: !tile_f32):
      %sum = "d2m.tile_add"(%lhs, %rhs) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %sum : !tile_f32
    } -> tensor<4x4x!tile_f32>
    d2m.yield %result : (tensor<4x4x!tile_f32>)
  } : tensor<4x4x!tile_f32>
  %3 = d2m.mask %2, %out3 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  return %3 : tensor<4x4x!tile_f32>
}

// CHECK-LABEL: func.func @keep_generic_add_one_one_result_mask
func.func @keep_generic_add_one_one_result_mask(
  %arg0: tensor<4x4x!tile_f32>,
  %arg1: tensor<4x4x!tile_f32>
) -> tensor<4x4x!tile_f32> {
  %out0 = d2m.empty() : tensor<4x4x!tile_f32>
  %out1 = d2m.empty() : tensor<4x4x!tile_f32>
  %out2 = d2m.empty() : tensor<4x4x!tile_f32>
  %out3 = d2m.empty() : tensor<4x4x!tile_f32>
  // CHECK: d2m.mask {{.*}} fill_value = <one>
  // CHECK: d2m.mask {{.*}} fill_value = <one>
  %0 = d2m.mask %arg0, %out0 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  %1 = d2m.mask %arg1, %out1 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  // CHECK: d2m.generic
  // CHECK: d2m.mask {{.*}} fill_value = <one>
  // CHECK: return
  %2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]}
      ins(%0, %1 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>)
      outs(%out2 : tensor<4x4x!tile_f32>) {
  ^unified0:
    %result = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%0, %1 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>) outs(%out2 : tensor<4x4x!tile_f32>) {
    ^bb0(%lhs: !tile_f32, %rhs: !tile_f32, %out: !tile_f32):
      %sum = "d2m.tile_add"(%lhs, %rhs) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %sum : !tile_f32
    } -> tensor<4x4x!tile_f32>
    d2m.yield %result : (tensor<4x4x!tile_f32>)
  } : tensor<4x4x!tile_f32>
  %3 = d2m.mask %2, %out3 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  return %3 : tensor<4x4x!tile_f32>
}

// CHECK-LABEL: func.func @keep_generic_mul_zero_untracked_result_mask
func.func @keep_generic_mul_zero_untracked_result_mask(
  %arg0: tensor<4x4x!tile_f32>,
  %arg1: tensor<4x4x!tile_f32>
) -> tensor<4x4x!tile_f32> {
  %out0 = d2m.empty() : tensor<4x4x!tile_f32>
  %out1 = d2m.empty() : tensor<4x4x!tile_f32>
  %out2 = d2m.empty() : tensor<4x4x!tile_f32>
  // CHECK: d2m.mask {{.*}} fill_value = <zero>
  %0 = d2m.mask %arg0, %out0 logical_shape = [50, 50] fill_value = <zero> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  // CHECK: d2m.generic
  // CHECK: d2m.tile_mul
  // CHECK: d2m.mask {{.*}} fill_value = <zero>
  // CHECK: return
  %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]}
      ins(%0, %arg1 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>)
      outs(%out1 : tensor<4x4x!tile_f32>) {
  ^unified0:
    %result = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%0, %arg1 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>) outs(%out1 : tensor<4x4x!tile_f32>) {
    ^bb0(%lhs: !tile_f32, %rhs: !tile_f32, %out: !tile_f32):
      %product = "d2m.tile_mul"(%lhs, %rhs) : (!tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %product : !tile_f32
    } -> tensor<4x4x!tile_f32>
    d2m.yield %result : (tensor<4x4x!tile_f32>)
  } : tensor<4x4x!tile_f32>
  %2 = d2m.mask %1, %out2 logical_shape = [50, 50] fill_value = <zero> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  return %2 : tensor<4x4x!tile_f32>
}

// CHECK-LABEL: func.func @remove_generic_add_scalar_zero_result_mask
func.func @remove_generic_add_scalar_zero_result_mask(
  %arg0: tensor<4x4x!tile_f32>
) -> tensor<4x4x!tile_f32> {
  %out0 = d2m.empty() : tensor<4x4x!tile_f32>
  %out1 = d2m.empty() : tensor<4x4x!tile_f32>
  %out2 = d2m.empty() : tensor<4x4x!tile_f32>
  // CHECK: d2m.mask {{.*}} fill_value = <one>
  %0 = d2m.mask %arg0, %out0 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  // CHECK: d2m.generic
  // CHECK-NOT: d2m.mask
  // CHECK: return
  %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]}
      ins(%0 : tensor<4x4x!tile_f32>)
      outs(%out1 : tensor<4x4x!tile_f32>) {
  ^unified0:
    %result = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%0 : tensor<4x4x!tile_f32>) outs(%out1 : tensor<4x4x!tile_f32>) {
    ^bb0(%input: !tile_f32, %out: !tile_f32):
      %zero = arith.constant 0.0 : f32
      %sum = "d2m.tile_add"(%input, %zero) : (!tile_f32, f32) -> !tile_f32
      linalg.yield %sum : !tile_f32
    } -> tensor<4x4x!tile_f32>
    d2m.yield %result : (tensor<4x4x!tile_f32>)
  } : tensor<4x4x!tile_f32>
  %2 = d2m.mask %1, %out2 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  return %2 : tensor<4x4x!tile_f32>
}

// CHECK-LABEL: func.func @remove_generic_exp_zero_result_mask
func.func @remove_generic_exp_zero_result_mask(
  %arg0: tensor<4x4x!tile_f32>
) -> tensor<4x4x!tile_f32> {
  %out0 = d2m.empty() : tensor<4x4x!tile_f32>
  %out1 = d2m.empty() : tensor<4x4x!tile_f32>
  %out2 = d2m.empty() : tensor<4x4x!tile_f32>
  // CHECK: d2m.mask {{.*}} fill_value = <zero>
  %0 = d2m.mask %arg0, %out0 logical_shape = [50, 50] fill_value = <zero> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  // CHECK: d2m.generic
  // CHECK: d2m.tile_exp
  // CHECK-NOT: d2m.mask
  // CHECK: return
  %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]}
      ins(%0 : tensor<4x4x!tile_f32>)
      outs(%out1 : tensor<4x4x!tile_f32>) {
  ^unified0:
    %result = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%0 : tensor<4x4x!tile_f32>) outs(%out1 : tensor<4x4x!tile_f32>) {
    ^bb0(%input: !tile_f32, %out: !tile_f32):
      %exp = "d2m.tile_exp"(%input) : (!tile_f32) -> !tile_f32
      linalg.yield %exp : !tile_f32
    } -> tensor<4x4x!tile_f32>
    d2m.yield %result : (tensor<4x4x!tile_f32>)
  } : tensor<4x4x!tile_f32>
  %2 = d2m.mask %1, %out2 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  return %2 : tensor<4x4x!tile_f32>
}

// CHECK-LABEL: func.func @remove_generic_eqz_zero_result_mask
func.func @remove_generic_eqz_zero_result_mask(
  %arg0: tensor<4x4x!tile_f32>
) -> tensor<4x4x!tile_f32> {
  %out0 = d2m.empty() : tensor<4x4x!tile_f32>
  %out1 = d2m.empty() : tensor<4x4x!tile_f32>
  %out2 = d2m.empty() : tensor<4x4x!tile_f32>
  // CHECK: d2m.mask {{.*}} fill_value = <zero>
  %0 = d2m.mask %arg0, %out0 logical_shape = [50, 50] fill_value = <zero> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  // CHECK: d2m.generic
  // CHECK: d2m.tile_eqz
  // CHECK-NOT: d2m.mask
  // CHECK: return
  %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]}
      ins(%0 : tensor<4x4x!tile_f32>)
      outs(%out1 : tensor<4x4x!tile_f32>) {
  ^unified0:
    %result = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%0 : tensor<4x4x!tile_f32>) outs(%out1 : tensor<4x4x!tile_f32>) {
    ^bb0(%input: !tile_f32, %out: !tile_f32):
      %eqz = "d2m.tile_eqz"(%input) : (!tile_f32) -> !tile_f32
      linalg.yield %eqz : !tile_f32
    } -> tensor<4x4x!tile_f32>
    d2m.yield %result : (tensor<4x4x!tile_f32>)
  } : tensor<4x4x!tile_f32>
  %2 = d2m.mask %1, %out2 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  return %2 : tensor<4x4x!tile_f32>
}

// CHECK-LABEL: func.func @remove_generic_reduce_sum_zero_result_mask
func.func @remove_generic_reduce_sum_zero_result_mask(
  %arg0: tensor<4x4x!tile_f32>,
  %arg1: tensor<4x4x!tile_f32>,
  %arg2: tensor<4x4x!tile_f32>
) -> tensor<4x4x!tile_f32> {
  %out0 = d2m.empty() : tensor<4x4x!tile_f32>
  %out1 = d2m.empty() : tensor<4x4x!tile_f32>
  %out2 = d2m.empty() : tensor<4x4x!tile_f32>
  %out3 = d2m.empty() : tensor<4x4x!tile_f32>
  %out4 = d2m.empty() : tensor<4x4x!tile_f32>
  // CHECK: d2m.mask {{.*}} fill_value = <zero>
  // CHECK: d2m.mask {{.*}} fill_value = <one>
  // CHECK: d2m.mask {{.*}} fill_value = <zero>
  %0 = d2m.mask %arg0, %out0 logical_shape = [50, 50] fill_value = <zero> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  %1 = d2m.mask %arg1, %out1 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  %2 = d2m.mask %arg2, %out2 logical_shape = [50, 50] fill_value = <zero> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  // CHECK: d2m.generic
  // CHECK: d2m.tile_reduce_sum
  // CHECK-NOT: d2m.mask
  // CHECK: return
  %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map, #map, #map], iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]}
      ins(%0, %1, %2 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>)
      outs(%out3 : tensor<4x4x!tile_f32>) {
  ^unified0:
    %result = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%0, %1, %2 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>) outs(%out3 : tensor<4x4x!tile_f32>) {
    ^bb0(%a: !tile_f32, %b: !tile_f32, %c: !tile_f32, %out: !tile_f32):
      %sum = "d2m.tile_reduce_sum"(%a, %b, %c) <{reduce_dim = #d2m<reduce_dim R>}> : (!tile_f32, !tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %sum : !tile_f32
    } -> tensor<4x4x!tile_f32>
    d2m.yield %result : (tensor<4x4x!tile_f32>)
  } : tensor<4x4x!tile_f32>
  %4 = d2m.mask %3, %out4 logical_shape = [50, 50] fill_value = <zero> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  return %4 : tensor<4x4x!tile_f32>
}

// CHECK-LABEL: func.func @remove_generic_reduce_max_neginf_result_mask
func.func @remove_generic_reduce_max_neginf_result_mask(
  %arg0: tensor<4x4x!tile_f32>,
  %arg1: tensor<4x4x!tile_f32>,
  %arg2: tensor<4x4x!tile_f32>
) -> tensor<4x4x!tile_f32> {
  %out0 = d2m.empty() : tensor<4x4x!tile_f32>
  %out1 = d2m.empty() : tensor<4x4x!tile_f32>
  %out2 = d2m.empty() : tensor<4x4x!tile_f32>
  %out3 = d2m.empty() : tensor<4x4x!tile_f32>
  %out4 = d2m.empty() : tensor<4x4x!tile_f32>
  // CHECK: d2m.mask {{.*}} fill_value = <neginf>
  // CHECK: d2m.mask {{.*}} fill_value = <one>
  // CHECK: d2m.mask {{.*}} fill_value = <neginf>
  %0 = d2m.mask %arg0, %out0 logical_shape = [50, 50] fill_value = <neginf> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  %1 = d2m.mask %arg1, %out1 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  %2 = d2m.mask %arg2, %out2 logical_shape = [50, 50] fill_value = <neginf> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  // CHECK: d2m.generic
  // CHECK: d2m.tile_reduce_max
  // CHECK-NOT: d2m.mask
  // CHECK: return
  %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map, #map, #map], iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]}
      ins(%0, %1, %2 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>)
      outs(%out3 : tensor<4x4x!tile_f32>) {
  ^unified0:
    %result = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%0, %1, %2 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>) outs(%out3 : tensor<4x4x!tile_f32>) {
    ^bb0(%a: !tile_f32, %b: !tile_f32, %c: !tile_f32, %out: !tile_f32):
      %max = "d2m.tile_reduce_max"(%a, %b, %c) <{reduce_dim = #d2m<reduce_dim C>}> : (!tile_f32, !tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %max : !tile_f32
    } -> tensor<4x4x!tile_f32>
    d2m.yield %result : (tensor<4x4x!tile_f32>)
  } : tensor<4x4x!tile_f32>
  %4 = d2m.mask %3, %out4 logical_shape = [50, 50] fill_value = <neginf> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  return %4 : tensor<4x4x!tile_f32>
}

// CHECK-LABEL: func.func @remove_generic_where_known_false_result_mask
func.func @remove_generic_where_known_false_result_mask(
  %arg0: tensor<4x4x!tile_f32>,
  %arg1: tensor<4x4x!tile_f32>,
  %arg2: tensor<4x4x!tile_f32>
) -> tensor<4x4x!tile_f32> {
  %out0 = d2m.empty() : tensor<4x4x!tile_f32>
  %out1 = d2m.empty() : tensor<4x4x!tile_f32>
  %out2 = d2m.empty() : tensor<4x4x!tile_f32>
  %out3 = d2m.empty() : tensor<4x4x!tile_f32>
  %out4 = d2m.empty() : tensor<4x4x!tile_f32>
  // CHECK: d2m.mask {{.*}} fill_value = <zero>
  // CHECK: d2m.mask {{.*}} fill_value = <one>
  // CHECK: d2m.mask {{.*}} fill_value = <zero>
  %0 = d2m.mask %arg0, %out0 logical_shape = [50, 50] fill_value = <zero> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  %1 = d2m.mask %arg1, %out1 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  %2 = d2m.mask %arg2, %out2 logical_shape = [50, 50] fill_value = <zero> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  // CHECK: d2m.generic
  // CHECK: d2m.tile_where
  // CHECK-NOT: d2m.mask
  // CHECK: return
  %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map, #map, #map], iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]}
      ins(%0, %1, %2 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>)
      outs(%out3 : tensor<4x4x!tile_f32>) {
  ^unified0:
    %result = linalg.generic {indexing_maps = [#map, #map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%0, %1, %2 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>) outs(%out3 : tensor<4x4x!tile_f32>) {
    ^bb0(%condition: !tile_f32, %true_value: !tile_f32, %false_value: !tile_f32, %out: !tile_f32):
      %where = "d2m.tile_where"(%condition, %true_value, %false_value) : (!tile_f32, !tile_f32, !tile_f32) -> !tile_f32
      linalg.yield %where : !tile_f32
    } -> tensor<4x4x!tile_f32>
    d2m.yield %result : (tensor<4x4x!tile_f32>)
  } : tensor<4x4x!tile_f32>
  %4 = d2m.mask %3, %out4 logical_shape = [50, 50] fill_value = <zero> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  return %4 : tensor<4x4x!tile_f32>
}

// CHECK-LABEL: func.func @remove_generic_mul_computed_scalar_one_result_mask
func.func @remove_generic_mul_computed_scalar_one_result_mask(
  %arg0: tensor<4x4x!tile_f32>
) -> tensor<4x4x!tile_f32> {
  %out0 = d2m.empty() : tensor<4x4x!tile_f32>
  %out1 = d2m.empty() : tensor<4x4x!tile_f32>
  %out2 = d2m.empty() : tensor<4x4x!tile_f32>
  // CHECK: d2m.mask {{.*}} fill_value = <one>
  %0 = d2m.mask %arg0, %out0 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  // CHECK: d2m.generic
  // CHECK: arith.addf
  // CHECK: d2m.tile_mul
  // CHECK-NOT: d2m.mask
  // CHECK: return
  %1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]}
      ins(%0 : tensor<4x4x!tile_f32>)
      outs(%out1 : tensor<4x4x!tile_f32>) {
  ^unified0:
    %result = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%0 : tensor<4x4x!tile_f32>) outs(%out1 : tensor<4x4x!tile_f32>) {
    ^bb0(%input: !tile_f32, %out: !tile_f32):
      %zero = arith.constant 0.0 : f32
      %one_const = arith.constant 1.0 : f32
      %one = arith.addf %zero, %one_const : f32
      %product = "d2m.tile_mul"(%input, %one) : (!tile_f32, f32) -> !tile_f32
      linalg.yield %product : !tile_f32
    } -> tensor<4x4x!tile_f32>
    d2m.yield %result : (tensor<4x4x!tile_f32>)
  } : tensor<4x4x!tile_f32>
  %2 = d2m.mask %1, %out2 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  return %2 : tensor<4x4x!tile_f32>
}

// CHECK-LABEL: func.func @remove_generic_logical_or_one_one_result_mask
func.func @remove_generic_logical_or_one_one_result_mask(
  %arg0: tensor<4x4x!tile_f32>,
  %arg1: tensor<4x4x!tile_f32>
) -> tensor<4x4x!tile_f32> {
  %out0 = d2m.empty() : tensor<4x4x!tile_f32>
  %out1 = d2m.empty() : tensor<4x4x!tile_f32>
  %out2 = d2m.empty() : tensor<4x4x!tile_f32>
  %out3 = d2m.empty() : tensor<4x4x!tile_f32>
  // CHECK: d2m.mask {{.*}} fill_value = <one>
  // CHECK: d2m.mask {{.*}} fill_value = <one>
  %0 = d2m.mask %arg0, %out0 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  %1 = d2m.mask %arg1, %out1 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  // CHECK: d2m.generic
  // CHECK: d2m.tile_nez
  // CHECK: d2m.tile_add
  // CHECK: d2m.tile_nez
  // CHECK-NOT: d2m.mask
  // CHECK: return
  %2 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
      indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel],
      threads = [#d2m.thread<unified>]}
      ins(%0, %1 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>)
      outs(%out2 : tensor<4x4x!tile_f32>) {
  ^unified0:
    %result = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]}
        ins(%0, %1 : tensor<4x4x!tile_f32>, tensor<4x4x!tile_f32>) outs(%out2 : tensor<4x4x!tile_f32>) {
    ^bb0(%lhs: !tile_f32, %rhs: !tile_f32, %out: !tile_f32):
      %lhs_nez = "d2m.tile_nez"(%lhs) : (!tile_f32) -> !tile_f32
      %rhs_nez = "d2m.tile_nez"(%rhs) : (!tile_f32) -> !tile_f32
      %sum = "d2m.tile_add"(%lhs_nez, %rhs_nez) : (!tile_f32, !tile_f32) -> !tile_f32
      %result_tile = "d2m.tile_nez"(%sum) : (!tile_f32) -> !tile_f32
      linalg.yield %result_tile : !tile_f32
    } -> tensor<4x4x!tile_f32>
    d2m.yield %result : (tensor<4x4x!tile_f32>)
  } : tensor<4x4x!tile_f32>
  %3 = d2m.mask %2, %out3 logical_shape = [50, 50] fill_value = <one> : tensor<4x4x!tile_f32> into tensor<4x4x!tile_f32> -> tensor<4x4x!tile_f32>
  return %3 : tensor<4x4x!tile_f32>
}
