// RUN: ttmlir-opt --d2m-scalarize-const-tensors %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#layout = #ttcore.metal_layout<logical_shape = 128x128, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded, index_map = map(0)>

!ttype_f32 = !ttcore.tile<32x32, f32>

module {
  // CHECK-LABEL: func.func @test_add_const_tensor_scalarize
  func.func @test_add_const_tensor_scalarize(%arg0: tensor<1x1x4x4x!ttype_f32, #layout>) -> tensor<1x1x4x4x!ttype_f32, #layout> {
    %cst = d2m.full {fill_value = 2.5 : f32, shape = array<i32: 128, 128>} : tensor<128x128xf32>
    %0 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    %1 = d2m.to_layout %cst, %0 : tensor<128x128xf32> into tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<1x1x4x4x!ttype_f32, #layout>
    %2 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>

    // CHECK: d2m.generic
    // CHECK: ins(%arg0 :
    // CHECK-NOT: ins(%arg0, %
    %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%arg0, %1 : tensor<1x1x4x4x!ttype_f32, #layout>, tensor<1x1x4x4x!ttype_f32, #layout>)
        outs(%2 : tensor<1x1x4x4x!ttype_f32, #layout>)  {
    ^compute0(%cb0: !d2m.cb<tensor<4x4x!ttype_f32>>, %cb1: !d2m.cb<tensor<4x4x!ttype_f32>>, %cb2: !d2m.cb<tensor<4x4x!ttype_f32>>):
      // CHECK: ^{{.*}}(%[[CB0:.*]]: !d2m.cb<tensor<4x4x!ttcore.tile<32x32, f32>>>, %[[CB_OUT:.*]]: !d2m.cb<tensor<4x4x!ttcore.tile<32x32, f32>>>):
      // CHECK-NOT: %cb1
      // CHECK: %[[SCALAR:.*]] = arith.constant 2.500000e+00 : f32
      %10 = d2m.wait %cb0 : <tensor<4x4x!ttype_f32>> -> tensor<4x4x!ttype_f32>
      %11 = d2m.wait %cb1 : <tensor<4x4x!ttype_f32>> -> tensor<4x4x!ttype_f32>
      // CHECK: %[[WAIT:.*]] = d2m.wait %[[CB0]]
      // CHECK-NOT: d2m.wait
      %12 = d2m.reserve %cb2 : <tensor<4x4x!ttype_f32>> -> tensor<4x4x!ttype_f32>
      // CHECK: %[[RESERVE:.*]] = d2m.reserve %[[CB_OUT]]
      %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %11 : tensor<4x4x!ttype_f32>, tensor<4x4x!ttype_f32>) outs(%12 : tensor<4x4x!ttype_f32>) {
      ^bb0(%in: !ttype_f32, %in_0: !ttype_f32, %out: !ttype_f32):
        // CHECK: linalg.generic {{.*}} ins(%[[WAIT]] :
        // CHECK-NOT: ins(%[[WAIT]], %
        // CHECK: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
        // CHECK-NOT: %in_0
        // CHECK: "d2m.tile_add"(%[[IN]], %[[SCALAR]]) : (!ttcore.tile<32x32, f32>, f32)
        %14 = "d2m.tile_add"(%in, %in_0) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        linalg.yield %14 : !ttype_f32
      } -> tensor<4x4x!ttype_f32>
      d2m.yield %13 : (tensor<4x4x!ttype_f32>)
    } : tensor<1x1x4x4x!ttype_f32, #layout>
    return %3 : tensor<1x1x4x4x!ttype_f32, #layout>
  }

  // CHECK-LABEL: func.func @test_mul_const_tensor_scalarize
  func.func @test_mul_const_tensor_scalarize(%arg0: tensor<1x1x4x4x!ttype_f32, #layout>) -> tensor<1x1x4x4x!ttype_f32, #layout> {
    %cst = d2m.full {fill_value = 3.0 : f32, shape = array<i32: 128, 128>} : tensor<128x128xf32>
    %0 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    %1 = d2m.to_layout %cst, %0 : tensor<128x128xf32> into tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<1x1x4x4x!ttype_f32, #layout>
    %2 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>

    // CHECK: d2m.generic
    // CHECK: ins(%arg0 :
    // CHECK-NOT: ins(%arg0, %
    %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%arg0, %1 : tensor<1x1x4x4x!ttype_f32, #layout>, tensor<1x1x4x4x!ttype_f32, #layout>)
        outs(%2 : tensor<1x1x4x4x!ttype_f32, #layout>)  {
    ^compute0(%cb0: !d2m.cb<tensor<4x4x!ttype_f32>>, %cb1: !d2m.cb<tensor<4x4x!ttype_f32>>, %cb2: !d2m.cb<tensor<4x4x!ttype_f32>>):
      // CHECK: ^{{.*}}(%[[CB0:.*]]: !d2m.cb<tensor<4x4x!ttcore.tile<32x32, f32>>>, %[[CB_OUT:.*]]: !d2m.cb<tensor<4x4x!ttcore.tile<32x32, f32>>>):
      // CHECK-NOT: %cb1
      // CHECK: %[[SCALAR:.*]] = arith.constant 3.000000e+00 : f32
      %10 = d2m.wait %cb0 : <tensor<4x4x!ttype_f32>> -> tensor<4x4x!ttype_f32>
      %11 = d2m.wait %cb1 : <tensor<4x4x!ttype_f32>> -> tensor<4x4x!ttype_f32>
      // CHECK: %[[WAIT:.*]] = d2m.wait %[[CB0]]
      // CHECK-NOT: d2m.wait
      %12 = d2m.reserve %cb2 : <tensor<4x4x!ttype_f32>> -> tensor<4x4x!ttype_f32>
      // CHECK: %[[RESERVE:.*]] = d2m.reserve %[[CB_OUT]]
      %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %11 : tensor<4x4x!ttype_f32>, tensor<4x4x!ttype_f32>) outs(%12 : tensor<4x4x!ttype_f32>) {
      ^bb0(%in: !ttype_f32, %in_0: !ttype_f32, %out: !ttype_f32):
        // CHECK: linalg.generic {{.*}} ins(%[[WAIT]] :
        // CHECK-NOT: ins(%[[WAIT]], %
        // CHECK: ^bb0(%[[IN:.*]]: !ttcore.tile<32x32, f32>, %[[OUT:.*]]: !ttcore.tile<32x32, f32>):
        // CHECK-NOT: %in_0
        // CHECK: "d2m.tile_mul"(%[[IN]], %[[SCALAR]]) : (!ttcore.tile<32x32, f32>, f32)
        %14 = "d2m.tile_mul"(%in, %in_0) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        linalg.yield %14 : !ttype_f32
      } -> tensor<4x4x!ttype_f32>
      d2m.yield %13 : (tensor<4x4x!ttype_f32>)
    } : tensor<1x1x4x4x!ttype_f32, #layout>
    return %3 : tensor<1x1x4x4x!ttype_f32, #layout>
  }

  // CHECK-LABEL: func.func @test_non_scalar_op_no_change
  // Test that operations that don't support scalars are not modified
  func.func @test_non_scalar_op_no_change(%arg0: tensor<1x1x4x4x!ttype_f32, #layout>) -> tensor<1x1x4x4x!ttype_f32, #layout> {
    %cst = d2m.full {fill_value = 2.5 : f32, shape = array<i32: 128, 128>} : tensor<128x128xf32>
    %0 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>
    %1 = d2m.to_layout %cst, %0 : tensor<128x128xf32> into tensor<1x1x4x4x!ttype_f32, #layout> -> tensor<1x1x4x4x!ttype_f32, #layout>
    %2 = d2m.empty() : tensor<1x1x4x4x!ttype_f32, #layout>

    // CHECK: d2m.generic
    %3 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%arg0, %1 : tensor<1x1x4x4x!ttype_f32, #layout>, tensor<1x1x4x4x!ttype_f32, #layout>)
        outs(%2 : tensor<1x1x4x4x!ttype_f32, #layout>)  {
    ^compute0(%cb0: !d2m.cb<tensor<4x4x!ttype_f32>>, %cb1: !d2m.cb<tensor<4x4x!ttype_f32>>, %cb2: !d2m.cb<tensor<4x4x!ttype_f32>>):
      %10 = d2m.wait %cb0 : <tensor<4x4x!ttype_f32>> -> tensor<4x4x!ttype_f32>
      %11 = d2m.wait %cb1 : <tensor<4x4x!ttype_f32>> -> tensor<4x4x!ttype_f32>
      %12 = d2m.reserve %cb2 : <tensor<4x4x!ttype_f32>> -> tensor<4x4x!ttype_f32>
      %13 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%10, %11 : tensor<4x4x!ttype_f32>, tensor<4x4x!ttype_f32>) outs(%12 : tensor<4x4x!ttype_f32>) {
      ^bb0(%in: !ttype_f32, %in_0: !ttype_f32, %out: !ttype_f32):
        // Use an operation that doesn't support scalars (e.g., maximum)
        // CHECK: "d2m.tile_maximum"(%{{.*}}, %{{.*}}) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>)
        // CHECK-NOT: arith.constant
        %14 = "d2m.tile_maximum"(%in, %in_0) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        linalg.yield %14 : !ttype_f32
      } -> tensor<4x4x!ttype_f32>
      d2m.yield %13 : (tensor<4x4x!ttype_f32>)
    } : tensor<1x1x4x4x!ttype_f32, #layout>
    return %3 : tensor<1x1x4x4x!ttype_f32, #layout>
  }
}
