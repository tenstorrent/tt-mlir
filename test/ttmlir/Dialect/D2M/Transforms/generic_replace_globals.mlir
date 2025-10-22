// RUN: ttmlir-opt --d2m-generic-replace-globals --split-input-file -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>

// -----

// Test basic replacement of globals within d2m.generic operations
module {
  // Define globals with indices
  ttcore.global @global_0 {index = 0 : i32} : memref<2x2xf32>
  ttcore.global @global_1 {index = 1 : i32} : memref<2x2xf32>

  // CHECK-LABEL: func.func @test_replace_globals
  func.func @test_replace_globals(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>, %arg2: memref<2x2xf32>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%arg0, %arg1 : memref<2x2xf32>, memref<2x2xf32>)
        outs(%arg2 : memref<2x2xf32>) {
    ^compute0(%cb0: memref<2x2xf32>, %cb1: memref<2x2xf32>, %cb2: memref<2x2xf32>):
      // CHECK-NOT: ttcore.get_global
      // CHECK: %[[GLOBAL0:.*]] = ttcore.get_global @global_0 : memref<2x2xf32>
      %global0 = ttcore.get_global @global_0 : memref<2x2xf32>
      // CHECK-NOT: ttcore.get_global
      // CHECK: %[[GLOBAL1:.*]] = ttcore.get_global @global_1 : memref<2x2xf32>
      %global1 = ttcore.get_global @global_1 : memref<2x2xf32>
      
      // The globals should be replaced with the corresponding operands
      // CHECK: %[[RESULT:.*]] = arith.addf %[[GLOBAL0]], %[[GLOBAL1]]
      %result = arith.addf %global0, %global1 : f32
      
      // CHECK: memref.store %[[RESULT]], %cb2[%c0, %c0] : memref<2x2xf32>
      %c0 = arith.constant 0 : index
      memref.store %result, %cb2[%c0, %c0] : memref<2x2xf32>
    }
    return
  }

  // CHECK-LABEL: func.func @test_no_replacement_outside_generic
  func.func @test_no_replacement_outside_generic() -> memref<2x2xf32> {
    // This get_global is outside a d2m.generic, so it should NOT be replaced
    // CHECK: %[[GLOBAL:.*]] = ttcore.get_global @global_0 : memref<2x2xf32>
    %0 = ttcore.get_global @global_0 : memref<2x2xf32>
    return %0 : memref<2x2xf32>
  }
}

// -----

// Test more complex replacement with multiple globals and complex computation
module {
  // Define globals with specific indices
  ttcore.global @input_global {index = 0 : i32} : memref<2x2xf32>
  ttcore.global @weight_global {index = 1 : i32} : memref<2x2xf32>
  ttcore.global @bias_global {index = 2 : i32} : memref<2x2xf32>

  // CHECK-LABEL: func.func @test_comprehensive_replacement
  func.func @test_comprehensive_replacement(%input: memref<2x2xf32>, %weight: memref<2x2xf32>, %bias: memref<2x2xf32>, %output: memref<2x2xf32>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%input, %weight, %bias : memref<2x2xf32>, memref<2x2xf32>, memref<2x2xf32>)
        outs(%output : memref<2x2xf32>) {
    ^compute0(%cb0: memref<2x2xf32>, %cb1: memref<2x2xf32>, %cb2: memref<2x2xf32>, %cb3: memref<2x2xf32>):
      // These get_global operations should be replaced with the corresponding operands
      // CHECK-NOT: ttcore.get_global @input_global
      // CHECK-NOT: ttcore.get_global @weight_global  
      // CHECK-NOT: ttcore.get_global @bias_global
      
      // The globals should be replaced with %cb0, %cb1, %cb2 respectively
      %input_val = ttcore.get_global @input_global : memref<2x2xf32>
      %weight_val = ttcore.get_global @weight_global : memref<2x2xf32>
      %bias_val = ttcore.get_global @bias_global : memref<2x2xf32>
      
      // Simulate some computation using the globals
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      
      // Load from the replaced operands
      %input_loaded = memref.load %input_val[%c0, %c0] : memref<2x2xf32>
      %weight_loaded = memref.load %weight_val[%c0, %c0] : memref<2x2xf32>
      %bias_loaded = memref.load %bias_val[%c0, %c0] : memref<2x2xf32>
      
      // Perform computation
      %mult_result = arith.mulf %input_loaded, %weight_loaded : f32
      %add_result = arith.addf %mult_result, %bias_loaded : f32
      
      // Store result
      memref.store %add_result, %cb3[%c0, %c0] : memref<2x2xf32>
    }
    return
  }

  // CHECK-LABEL: func.func @test_mixed_usage
  func.func @test_mixed_usage(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
    // This get_global is outside a d2m.generic, so it should remain unchanged
    // CHECK: %[[OUTSIDE_GLOBAL:.*]] = ttcore.get_global @input_global : memref<2x2xf32>
    %outside_global = ttcore.get_global @input_global : memref<2x2xf32>
    
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%arg0 : memref<2x2xf32>)
        outs(%arg1 : memref<2x2xf32>) {
    ^compute0(%cb0: memref<2x2xf32>, %cb1: memref<2x2xf32>):
      // This get_global is inside a d2m.generic, so it should be replaced
      // CHECK-NOT: ttcore.get_global @input_global
      %inside_global = ttcore.get_global @input_global : memref<2x2xf32>
      
      %c0 = arith.constant 0 : index
      %val = memref.load %inside_global[%c0, %c0] : memref<2x2xf32>
      memref.store %val, %cb1[%c0, %c0] : memref<2x2xf32>
    }
    return
  }
}
