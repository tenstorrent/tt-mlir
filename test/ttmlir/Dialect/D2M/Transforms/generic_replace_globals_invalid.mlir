// RUN: not ttmlir-opt --d2m-generic-replace-globals -o %t %s 2>&1 | FileCheck %s

#l1_ = #ttcore.memory_space<l1>

module {
  // Test case 1: Global without index attribute
  ttcore.global @global_no_index : memref<2x2xf32>

  // Test case 2: Global with index but referencing non-existent global
  ttcore.global @global_with_index {index = 0 : i32} : memref<2x2xf32>

  // CHECK-LABEL: func.func @test_missing_index
  func.func @test_missing_index(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%arg0 : memref<2x2xf32>)
        outs(%arg1 : memref<2x2xf32>) {
    ^compute0(%cb0: memref<2x2xf32>, %cb1: memref<2x2xf32>):
      // CHECK: error: Global must have a valid index attribute
      %global = ttcore.get_global @global_no_index : memref<2x2xf32>
    }
    return
  }

  // CHECK-LABEL: func.func @test_nonexistent_global
  func.func @test_nonexistent_global(%arg0: memref<2x2xf32>, %arg1: memref<2x2xf32>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%arg0 : memref<2x2xf32>)
        outs(%arg1 : memref<2x2xf32>) {
    ^compute0(%cb0: memref<2x2xf32>, %cb1: memref<2x2xf32>):
      // CHECK: error: Global symbol not found: @nonexistent_global
      %global = ttcore.get_global @nonexistent_global : memref<2x2xf32>
    }
    return
  }
}
