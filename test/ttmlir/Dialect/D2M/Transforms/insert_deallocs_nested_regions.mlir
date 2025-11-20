// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-deallocs -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that D2MInsertDeallocs correctly handles nested regions and
// recursively identifies memref uses inside d2m.generic, linalg.generic, 
// scf.for, and other operations with nested regions.

#l1 = #ttcore.memory_space<l1>

// CHECK-LABEL: func.func @nested_linalg_use
func.func @nested_linalg_use() -> memref<16x16x!ttcore.tile<32x32, f32>, #l1> {
  // Test that buffer used inside nested linalg.generic is kept alive
  
  // CHECK: %[[INPUT:.*]] = memref.alloc(){{.*}}#l1>
  %input = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // CHECK: %[[TEMP:.*]] = memref.alloc(){{.*}}#l1>
  %temp = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // CHECK: %[[RESULT:.*]] = memref.alloc(){{.*}}#l1>
  %result = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // Dealloc should NOT appear before d2m.generic completes
  // CHECK-NOT: memref.dealloc %[[INPUT]]
  
  // CHECK: d2m.generic
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                affine_map<(d0, d1) -> (d0, d1)>,
                                affine_map<(d0, d1) -> (d0, d1)>],
               iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
               threads = [#d2m.thread<compute>]}
      ins(%input, %temp : memref<16x16x!ttcore.tile<32x32, f32>, #l1>,
                          memref<16x16x!ttcore.tile<32x32, f32>, #l1>)
      outs(%result : memref<16x16x!ttcore.tile<32x32, f32>, #l1>) {
  ^compute0(%cb_in: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_temp: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_out: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>):
    %0 = d2m.wait %cb_in : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    %1 = d2m.wait %cb_temp : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    %2 = d2m.reserve %cb_out : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    
    // Nested linalg.generic uses %0 (which comes from %input via cb_in)
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                     affine_map<(d0, d1) -> (d0, d1)>,
                                     affine_map<(d0, d1) -> (d0, d1)>],
                    iterator_types = ["parallel", "parallel"]}
        ins(%0, %1 : memref<16x16x!ttcore.tile<32x32, f32>, #l1>,
                     memref<16x16x!ttcore.tile<32x32, f32>, #l1>)
        outs(%2 : memref<16x16x!ttcore.tile<32x32, f32>, #l1>) {
    ^bb0(%arg0: !ttcore.tile<32x32, f32>, 
         %arg1: !ttcore.tile<32x32, f32>, 
         %arg2: !ttcore.tile<32x32, f32>):
      %3 = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %3 : !ttcore.tile<32x32, f32>
    }
  }
  // CHECK: }
  
  // Deallocs should appear AFTER d2m.generic completes
  // CHECK: memref.dealloc %[[TEMP]]
  // CHECK: memref.dealloc %[[INPUT]]
  
  return %result : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
}

// CHECK-LABEL: func.func @nested_scf_for_use
func.func @nested_scf_for_use() -> memref<16x16x!ttcore.tile<32x32, f32>, #l1> {
  // Test that buffer used inside scf.for loop (after linalg-to-affine conversion) is kept alive
  
  // CHECK: %[[INPUT:.*]] = memref.alloc(){{.*}}#l1>
  %input = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // CHECK: %[[RESULT:.*]] = memref.alloc(){{.*}}#l1>
  %result = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // Dealloc should NOT appear before d2m.generic completes
  // CHECK-NOT: memref.dealloc %[[INPUT]]
  
  // CHECK: d2m.generic
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                affine_map<(d0, d1) -> (d0, d1)>],
               iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
               threads = [#d2m.thread<compute>]}
      ins(%input : memref<16x16x!ttcore.tile<32x32, f32>, #l1>)
      outs(%result : memref<16x16x!ttcore.tile<32x32, f32>, #l1>) {
  ^compute0(%cb_in: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_out: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>):
    %0 = d2m.wait %cb_in : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    %1 = d2m.reserve %cb_out : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    
    %c0 = arith.constant 0 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    
    // Nested scf.for loop uses %0 (which comes from %input via cb_in)
    scf.for %i = %c0 to %c16 step %c1 {
      scf.for %j = %c0 to %c16 step %c1 {
        %val = affine.load %0[%i, %j] : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
        %exp = "d2m.tile_exp"(%val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %exp, %1[%i, %j] : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
      }
    }
  }
  // CHECK: }
  
  // Dealloc should appear AFTER d2m.generic completes
  // CHECK: memref.dealloc %[[INPUT]]
  
  return %result : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
}

// CHECK-LABEL: func.func @deeply_nested_use
func.func @deeply_nested_use() -> memref<16x16x!ttcore.tile<32x32, f32>, #l1> {
  // Test that buffer used in deeply nested structure is kept alive
  // d2m.generic -> scf.for -> scf.for -> affine.load
  
  // CHECK: %[[INPUT:.*]] = memref.alloc(){{.*}}#l1>
  %input = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // CHECK: %[[RESULT:.*]] = memref.alloc(){{.*}}#l1>
  %result = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // Dealloc should NOT appear before d2m.generic completes
  // CHECK-NOT: memref.dealloc %[[INPUT]]
  
  // CHECK: d2m.generic
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                affine_map<(d0, d1) -> (d0, d1)>],
               iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
               threads = [#d2m.thread<compute>]}
      ins(%input : memref<16x16x!ttcore.tile<32x32, f32>, #l1>)
      outs(%result : memref<16x16x!ttcore.tile<32x32, f32>, #l1>) {
  ^compute0(%cb_in: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_out: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>):
    %0 = d2m.wait %cb_in : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    %1 = d2m.reserve %cb_out : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    
    %c0 = arith.constant 0 : index
    %c8 = arith.constant 8 : index
    %c16 = arith.constant 16 : index
    %c1 = arith.constant 1 : index
    
    // Outer loop
    scf.for %i_outer = %c0 to %c16 step %c8 {
      // Middle loop
      scf.for %j_outer = %c0 to %c16 step %c8 {
        // Inner loops with actual use of %0
        scf.for %i = %c0 to %c8 step %c1 {
          scf.for %j = %c0 to %c8 step %c1 {
            %i_idx = arith.addi %i_outer, %i : index
            %j_idx = arith.addi %j_outer, %j : index
            %val = affine.load %0[%i_idx, %j_idx] : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
            affine.store %val, %1[%i_idx, %j_idx] : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
          }
        }
      }
    }
  }
  // CHECK: }
  
  // Dealloc should appear AFTER d2m.generic completes
  // CHECK: memref.dealloc %[[INPUT]]
  
  return %result : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
}

// CHECK-LABEL: func.func @multiple_nested_regions_same_buffer
func.func @multiple_nested_regions_same_buffer() -> memref<16x16x!ttcore.tile<32x32, f32>, #l1> {
  // Test that a buffer used in multiple nested regions is kept alive throughout
  
  // CHECK: %[[SHARED:.*]] = memref.alloc(){{.*}}#l1>
  %shared = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // CHECK: %[[TEMP:.*]] = memref.alloc(){{.*}}#l1>
  %temp = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // CHECK: %[[RESULT:.*]] = memref.alloc(){{.*}}#l1>
  %result = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // Dealloc should NOT appear before both d2m.generic regions complete
  // CHECK-NOT: memref.dealloc %[[SHARED]]
  
  // First d2m.generic uses shared buffer
  // CHECK: d2m.generic
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                affine_map<(d0, d1) -> (d0, d1)>],
               iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
               threads = [#d2m.thread<compute>]}
      ins(%shared : memref<16x16x!ttcore.tile<32x32, f32>, #l1>)
      outs(%temp : memref<16x16x!ttcore.tile<32x32, f32>, #l1>) {
  ^compute0(%cb_in: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_out: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>):
    %0 = d2m.wait %cb_in : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    %1 = d2m.reserve %cb_out : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                     affine_map<(d0, d1) -> (d0, d1)>],
                    iterator_types = ["parallel", "parallel"]}
        ins(%0 : memref<16x16x!ttcore.tile<32x32, f32>, #l1>)
        outs(%1 : memref<16x16x!ttcore.tile<32x32, f32>, #l1>) {
    ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
      %2 = "d2m.tile_exp"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %2 : !ttcore.tile<32x32, f32>
    }
  }
  // CHECK: }
  
  // Dealloc should still NOT appear - shared buffer used again below
  // CHECK-NOT: memref.dealloc %[[SHARED]]
  
  // Second d2m.generic also uses shared buffer
  // CHECK: d2m.generic
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                affine_map<(d0, d1) -> (d0, d1)>,
                                affine_map<(d0, d1) -> (d0, d1)>],
               iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
               threads = [#d2m.thread<compute>]}
      ins(%shared, %temp : memref<16x16x!ttcore.tile<32x32, f32>, #l1>,
                           memref<16x16x!ttcore.tile<32x32, f32>, #l1>)
      outs(%result : memref<16x16x!ttcore.tile<32x32, f32>, #l1>) {
  ^compute1(%cb_in1: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_in2: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_out: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>):
    %3 = d2m.wait %cb_in1 : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    %4 = d2m.wait %cb_in2 : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    %5 = d2m.reserve %cb_out : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                     affine_map<(d0, d1) -> (d0, d1)>,
                                     affine_map<(d0, d1) -> (d0, d1)>],
                    iterator_types = ["parallel", "parallel"]}
        ins(%3, %4 : memref<16x16x!ttcore.tile<32x32, f32>, #l1>,
                     memref<16x16x!ttcore.tile<32x32, f32>, #l1>)
        outs(%5 : memref<16x16x!ttcore.tile<32x32, f32>, #l1>) {
    ^bb0(%arg0: !ttcore.tile<32x32, f32>, 
         %arg1: !ttcore.tile<32x32, f32>, 
         %arg2: !ttcore.tile<32x32, f32>):
      %6 = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %6 : !ttcore.tile<32x32, f32>
    }
  }
  // CHECK: }
  
  // NOW deallocs should appear AFTER both d2m.generic regions complete
  // CHECK: memref.dealloc %[[TEMP]]
  // CHECK: memref.dealloc %[[SHARED]]
  
  return %result : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
}
