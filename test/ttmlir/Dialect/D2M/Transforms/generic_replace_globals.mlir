// RUN: ttmlir-opt --d2m-generic-replace-globals --split-input-file %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>

// -----

// Test basic replacement of globals within d2m.generic operations
module {
  // Define globals with indices
  ttcore.global @global_0 = tensor<2x2xf32> [0]
  ttcore.global @global_1 = tensor<2x2xf32> [1]

  // CHECK-LABEL: func.func @test_replace_globals
  func.func @test_replace_globals(%arg0: tensor<2x2xf32>, %arg1: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = d2m.empty() : tensor<2x2xf32>
    %1 = "d2m.generic"(%arg0, %arg1, %0) <{
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>],
      operandSegmentSizes = array<i32: 2, 1>}> ({
    ^bb0(%cb0: memref<2x2xf32>, %cb1: memref<2x2xf32>, %cb2: memref<2x2xf32>):
      // The globals should be replaced with the corresponding operands
      // Load values from the memrefs and perform computation
      %c0 = arith.constant 0 : index
      %val0 = memref.load %cb0[%c0, %c0] : memref<2x2xf32>
      %val1 = memref.load %cb1[%c0, %c0] : memref<2x2xf32>
      %result = arith.addf %val0, %val1 : f32

      // CHECK: memref.store %{{.*}}, %cb2[%c0, %c0] : memref<2x2xf32>
      memref.store %result, %cb2[%c0, %c0] : memref<2x2xf32>
      "d2m.yield"(%cb2) : (memref<2x2xf32>) -> ()
    }) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }

  // CHECK-LABEL: func.func @test_no_replacement_outside_generic
  func.func @test_no_replacement_outside_generic() -> tensor<2x2xf32> {
    // This get_global is outside a d2m.generic, so it should NOT be replaced
    // CHECK: %[[GLOBAL:.*]] = ttcore.get_global @global_0 : tensor<2x2xf32>
    %0 = ttcore.get_global @global_0 : tensor<2x2xf32>
    return %0 : tensor<2x2xf32>
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
  func.func @test_comprehensive_replacement(%input: tensor<2x2xf32>, %weight: tensor<2x2xf32>, %bias: tensor<2x2xf32>) -> tensor<2x2xf32> {
    %0 = d2m.empty() : tensor<2x2xf32>
    %1 = "d2m.generic"(%input, %weight, %bias, %0) <{
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>],
      operandSegmentSizes = array<i32: 3, 1>}> ({
    ^bb0(%cb0: memref<2x2xf32>, %cb1: memref<2x2xf32>, %cb2: memref<2x2xf32>, %cb3: memref<2x2xf32>):
      // These get_global operations should be replaced with the corresponding operands
      // The globals should be replaced with %cb0, %cb1, %cb2 respectively

      // Simulate some computation using the globals
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index

      // Load from the replaced operands
      %input_loaded = memref.load %cb0[%c0, %c0] : memref<2x2xf32>
      %weight_loaded = memref.load %cb1[%c0, %c0] : memref<2x2xf32>
      %bias_loaded = memref.load %cb2[%c0, %c0] : memref<2x2xf32>

      // Perform computation
      %mult_result = arith.mulf %input_loaded, %weight_loaded : f32
      %add_result = arith.addf %mult_result, %bias_loaded : f32

      // Store result
      memref.store %add_result, %cb3[%c0, %c0] : memref<2x2xf32>
      "d2m.yield"(%cb3) : (memref<2x2xf32>) -> ()
    }) : (tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %1 : tensor<2x2xf32>
  }

  // CHECK-LABEL: func.func @test_mixed_usage
  func.func @test_mixed_usage(%arg0: tensor<2x2xf32>) -> tensor<2x2xf32> {
    // This get_global is outside a d2m.generic, so it should remain unchanged
    // CHECK: %[[OUTSIDE_GLOBAL:.*]] = ttcore.get_global @input_global : tensor<2x2xf32>
    %outside_global = ttcore.get_global @input_global : tensor<2x2xf32>

    %0 = d2m.empty() : tensor<2x2xf32>
    %1 = "d2m.generic"(%arg0, %0) <{
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<compute>],
      operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%cb0: memref<2x2xf32>, %cb1: memref<2x2xf32>):
      // This get_global is inside a d2m.generic, so it should be replaced

      %c0 = arith.constant 0 : index
      %val = memref.load %cb0[%c0, %c0] : memref<2x2xf32>
      memref.store %val, %cb1[%c0, %c0] : memref<2x2xf32>
      "d2m.yield"(%cb1) : (memref<2x2xf32>) -> ()
    }) : (tensor<2x2xf32>, tensor<2x2xf32>) -> tensor<2x2xf32>
    return %outside_global : tensor<2x2xf32>
  }
}
