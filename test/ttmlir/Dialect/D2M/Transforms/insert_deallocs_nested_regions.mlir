// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-deallocs -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that D2MInsertDeallocs correctly handles nested regions and
// recursively identifies memref uses inside d2m.generic with nested affine loops.

#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK-LABEL: func.func @nested_affine_loops
func.func @nested_affine_loops() -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> {
  // Test that buffers used inside nested affine loops within d2m.generic are kept alive
  
  // CHECK: %[[INPUT:.*]] = memref.alloc()
  %input = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  
  // CHECK: %[[RESULT:.*]] = memref.alloc()
  %result = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  
  // Dealloc should NOT appear before d2m.generic completes
  // CHECK-NOT: memref.dealloc %[[INPUT]]
  
  // CHECK: d2m.generic
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [#map, #map],
               iterator_types = [#parallel, #parallel],
               threads = [#d2m.thread<compute>]}
      ins(%input : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
      outs(%result : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^compute0(%cb_in: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_out: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    %0 = d2m.wait %cb_in : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    %1 = d2m.reserve %cb_out : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    
    // Nested affine loops use %0 (which comes from %input via cb_in)
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 4 {
        %val = affine.load %0[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %exp = "d2m.tile_exp"(%val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %exp, %1[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      }
    }
  }
  // CHECK: }
  
  // Dealloc should appear AFTER d2m.generic completes
  // CHECK: memref.dealloc %[[INPUT]]
  
  return %result : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
}

// CHECK-LABEL: func.func @deeply_nested_loops
func.func @deeply_nested_loops() -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> {
  // Test that buffer used in deeply nested affine loops is kept alive
  
  // CHECK: %[[INPUT:.*]] = memref.alloc()
  %input = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  
  // CHECK: %[[RESULT:.*]] = memref.alloc()
  %result = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  
  // Dealloc should NOT appear before d2m.generic completes
  // CHECK-NOT: memref.dealloc %[[INPUT]]
  
  // CHECK: d2m.generic
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [#map, #map],
               iterator_types = [#parallel, #parallel],
               threads = [#d2m.thread<compute>]}
      ins(%input : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
      outs(%result : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^compute0(%cb_in: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_out: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    %0 = d2m.wait %cb_in : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    %1 = d2m.reserve %cb_out : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    
    // Deeply nested loops - outer → middle → inner
    affine.for %i_outer = 0 to 2 {
      affine.for %j_outer = 0 to 2 {
        affine.for %i_inner = 0 to 1 {
          affine.for %j_inner = 0 to 2 {
            %i_idx = affine.apply affine_map<(d0) -> (d0)>(%i_outer)
            %j_idx = affine.apply affine_map<(d0, d1) -> (d0 * 2 + d1)>(%j_outer, %j_inner)
            %val = affine.load %0[%i_idx, %j_idx] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
            affine.store %val, %1[%i_idx, %j_idx] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
          }
        }
      }
    }
  }
  // CHECK: }
  
  // Dealloc should appear AFTER d2m.generic completes
  // CHECK: memref.dealloc %[[INPUT]]
  
  return %result : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
}

// CHECK-LABEL: func.func @multiple_buffers_same_region
func.func @multiple_buffers_same_region() -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> {
  // Test that multiple buffers used in same region are all kept alive
  
  // CHECK: %[[INPUT_A:.*]] = memref.alloc()
  %input_a = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  // CHECK: %[[INPUT_B:.*]] = memref.alloc()
  %input_b = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  
  // CHECK: %[[RESULT:.*]] = memref.alloc()
  %result = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  
  // Neither input should be deallocated before d2m.generic
  // CHECK-NOT: memref.dealloc %[[INPUT_A]]
  // CHECK-NOT: memref.dealloc %[[INPUT_B]]
  
  // CHECK: d2m.generic
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [#map, #map, #map],
               iterator_types = [#parallel, #parallel],
               threads = [#d2m.thread<compute>]}
      ins(%input_a, %input_b : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>,
                               memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
      outs(%result : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^compute0(%cb_a: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_b: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_out: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    %0 = d2m.wait %cb_a : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    %1 = d2m.wait %cb_b : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    %2 = d2m.reserve %cb_out : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 4 {
        %a = affine.load %0[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %b = affine.load %1[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %sum = "d2m.tile_add"(%a, %b) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %sum, %2[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      }
    }
  }
  // CHECK: }
  
  // Both inputs should be deallocated AFTER d2m.generic completes
  // CHECK: memref.dealloc %[[INPUT_B]]
  // CHECK: memref.dealloc %[[INPUT_A]]
  
  return %result : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
}
