// RUN: ttmlir-opt --d2m-generic-replace-globals --split-input-file %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>

// -----

// Test global with get_global outside generic
module {
  ttcore.global @global_0 = tensor<2x2xf32> [0]
  // CHECK-LABEL: func.func @test_no_replacement_outside_generic
  func.func @test_no_replacement_outside_generic() -> tensor<2x2xf32> {
    // This get_global is outside a d2m.generic, so it should NOT be replaced
    // CHECK: %[[GLOBAL:.*]] = ttcore.get_global @global_0 : tensor<2x2xf32>
    %0 = ttcore.get_global @global_0 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
  }
}

// -----

// Test basic replacement of globals within d2m.generic operations
module {
  // Define globals with indices
  ttcore.global @global_0 = tensor<2x2xf32> [0]
  ttcore.global @global_1 = tensor<2x2xf32> [1]
  // CHECK-LABEL: func.func @test_replace_globals
  // CHECK-SAME: (%[[ARG0:.*]]: tensor<2x2xf32>, %[[ARG1:.*]]: tensor<2x2xf32>) -> tensor<2x2xf32>
  func.func @test_replace_globals(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = d2m.empty() : tensor<2x2xf32>
    // CHECK: d2m.generic
    %1 = d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]}
      ins(%arg0, %arg1 : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%0 : tensor<2x2xf32>) {
    ^bb0(%cb0: !d2m.cb<tensor<2x2xf32>>, %cb1: !d2m.cb<tensor<2x2xf32>>, %cb2: !d2m.cb<tensor<2x2xf32>>):
      // These get_global operations should be replaced with the corresponding operands
      %global0 = ttcore.get_global @global_0 : tensor<2x2xf32>
      %global1 = ttcore.get_global @global_1 : tensor<2x2xf32>
      // CHECK-NOT: ttcore.get_global
      // CHECK: d2m.wait %cb0
      %arg0_val = d2m.wait %cb0 : !d2m.cb<tensor<2x2xf32>> -> tensor<2x2xf32>
      // CHECK: d2m.wait %cb1
      %arg1_val = d2m.wait %cb1 : !d2m.cb<tensor<2x2xf32>> -> tensor<2x2xf32>
      // CHECK: d2m.reserve %cb2
      %arg2_val = d2m.reserve %cb2 : !d2m.cb<tensor<2x2xf32>> -> tensor<2x2xf32>
      // CHECK: linalg.generic
      // CHECK-SAME: ins(%[[ARG0]], %[[ARG1]] : tensor<2x2xf32>, tensor<2x2xf32>)
      // CHECK-SAME: outs(%{{.*}} : tensor<2x2xf32>)
      %result = linalg.generic {
        indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
        iterator_types = ["parallel", "parallel"]
      }
      ins(%global0, %global1 : tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%arg2_val : tensor<2x2xf32>) {
      ^bb1(%a: f32, %b: f32, %c: f32):
        %sum = arith.addf %a, %b : f32
        linalg.yield %sum : f32
      } -> tensor<2x2xf32>
      d2m.yield %result : (tensor<2x2xf32>)
    } : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
}

// -----

// Test more complex replacement with multiple globals and complex computation
module {
  // Define globals with specific indices
  ttcore.global @input_global = tensor<2x2xf32> [0]
  ttcore.global @weight_global = tensor<2x2xf32> [1]
  ttcore.global @bias_global = tensor<2x2xf32> [2]
  // CHECK-LABEL: func.func @test_comprehensive_replacement
  // CHECK-SAME: (%[[INPUT:.*]]: tensor<2x2xf32>, %[[WEIGHT:.*]]: tensor<2x2xf32>, %[[BIAS:.*]]: tensor<2x2xf32>) -> tensor<2x2xf32>
  func.func @test_comprehensive_replacement(%input: tensor<2x2xf32>, %weight: tensor<2x2xf32>, %bias: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = d2m.empty() : tensor<2x2xf32>
    // CHECK: d2m.generic
    %1 = d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]}
      ins(%input, %weight, %bias : tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>)
      outs(%0 : tensor<2x2xf32>) {
    ^bb0(%cb0: !d2m.cb<tensor<2x2xf32>>, %cb1: !d2m.cb<tensor<2x2xf32>>, %cb2: !d2m.cb<tensor<2x2xf32>>, %cb3: !d2m.cb<tensor<2x2xf32>>):
      // These get_global operations should be replaced with the corresponding operands
      %input_global = ttcore.get_global @input_global : tensor<2x2xf32>
      %weight_global = ttcore.get_global @weight_global : tensor<2x2xf32>
      %bias_global = ttcore.get_global @bias_global : tensor<2x2xf32>
      // CHECK-NOT: ttcore.get_global
      // CHECK: d2m.wait %cb0
      %input_val = d2m.wait %cb0 : !d2m.cb<tensor<2x2xf32>> -> tensor<2x2xf32>
      // CHECK: d2m.wait %cb1
      %weight_val = d2m.wait %cb1 : !d2m.cb<tensor<2x2xf32>> -> tensor<2x2xf32>
      // CHECK: d2m.wait %cb2
      %bias_val = d2m.wait %cb2 : !d2m.cb<tensor<2x2xf32>> -> tensor<2x2xf32>
      // CHECK: [[OUTPUT_VAL:%.*]] = d2m.reserve %cb3
      %output_val = d2m.reserve %cb3 : !d2m.cb<tensor<2x2xf32>> -> tensor<2x2xf32>
      // Simulate some computation using the globals
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      // Load from the replaced operands
      // CHECK: tensor.extract %[[INPUT]][%c0, %c0] : tensor<2x2xf32>
      %input_scalar = tensor.extract %input_global[%c0, %c0] : tensor<2x2xf32>
      // CHECK: tensor.extract %[[WEIGHT]][%c0, %c0] : tensor<2x2xf32>
      %weight_scalar = tensor.extract %weight_global[%c0, %c0] : tensor<2x2xf32>
      // CHECK: tensor.extract %[[BIAS]][%c0, %c0] : tensor<2x2xf32>
      %bias_scalar = tensor.extract %bias_global[%c0, %c0] : tensor<2x2xf32>
      // Perform computation
      %mult_result = arith.mulf %input_scalar, %weight_scalar : f32
      // CHECK: [[ADD_RESULT:%.*]] = arith.addf
      %add_result = arith.addf %mult_result, %bias_scalar : f32
      // CHECK: tensor.insert [[ADD_RESULT]] into [[OUTPUT_VAL]][%c0, %c0] : tensor<2x2xf32>
      %result_tensor = tensor.insert %add_result into %output_val[%c0, %c0] : tensor<2x2xf32>
      d2m.yield %result_tensor : (tensor<2x2xf32>)
    } : tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }
  // CHECK-LABEL: func.func @test_mixed_usage
  // CHECK-SAME:     (%[[ARG0:.*]]: tensor<2x2xf32>) -> tensor<2x2xf32>
  func.func @test_mixed_usage(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    // This get_global is outside a d2m.generic, so it should remain unchanged
    // CHECK: %[[OUTSIDE_GLOBAL:.*]] = ttcore.get_global @input_global : tensor<2x2xf32>
    %outside_global = ttcore.get_global @input_global : tensor<2x2xf32>
    %0 = d2m.empty() : tensor<2x2xf32>
    // CHECK: d2m.generic
    %1 = d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>]}
      ins(%arg0 : tensor<2x2xf32>)
      outs(%0 : tensor<2x2xf32>) {
    ^bb0(%cb0: !d2m.cb<tensor<2x2xf32>>, %cb1: !d2m.cb<tensor<2x2xf32>>):
      // This get_global is inside a d2m.generic, so it should be replaced
      %input_global = ttcore.get_global @input_global : tensor<2x2xf32>
      // CHECK-NOT: ttcore.get_global
      // CHECK: d2m.wait %cb0
      %arg0_val = d2m.wait %cb0 : !d2m.cb<tensor<2x2xf32>> -> tensor<2x2xf32>
      // CHECK: d2m.reserve %cb1
      %arg1_val = d2m.reserve %cb1 : !d2m.cb<tensor<2x2xf32>> -> tensor<2x2xf32>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      // CHECK: scf.for
      %result = scf.for %i = %c0 to %c2 step %c1 iter_args(%iter_arg = %arg1_val) -> (tensor<2x2xf32>) {
        // CHECK: tensor.extract %[[ARG0]][%{{.*}}, %{{.*}}] : tensor<2x2xf32>
        %val = tensor.extract %input_global[%i, %i] : tensor<2x2xf32>
        // CHECK: tensor.insert %{{.*}} into %{{.*}}[%{{.*}}, %{{.*}}] : tensor<2x2xf32>
        %inserted = tensor.insert %val into %iter_arg[%i, %i] : tensor<2x2xf32>
        scf.yield %inserted : tensor<2x2xf32>
      }
      d2m.yield %result : (tensor<2x2xf32>)
    } : tensor<2x2xf32>
    return %outside_global : tensor<2x2xf32>
  }
}
