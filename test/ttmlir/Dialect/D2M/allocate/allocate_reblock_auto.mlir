//// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=test-assume-l1-capacity=8388608 test-buffer-size-policy=max" -o %t.max %s
//// RUN: FileCheck %s --check-prefix=CHECK-MAX --input-file=%t.max
//// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=test-assume-l1-capacity=8388608" -o %t.auto %s
//// RUN: FileCheck %s --check-prefix=CHECK-AUTO --input-file=%t.auto
//
//#l1 = #ttcore.memory_space<l1>
//#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
//#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
//#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>
//#broadcast = affine_map<(d0, d1) -> (0, 0)>
//#reduceIn = affine_map<(d0, d1) -> (d0, d1)>
//#reduceOut = affine_map<(d0, d1) -> (d0, 0)>
//#multiReduceIn = affine_map<(d0, d1) -> (d0, d1)>
//#multiReduceOut = affine_map<(d0, d1) -> (0, 0)>
//#parallel = #ttcore.iterator_type<parallel>
//#reduction = #ttcore.iterator_type<reduction>
//
//module {
//  // CHECK-MAX-LABEL: func.func @matmul_auto_vs_max()
//  // CHECK-MAX: memref.alloc() {{.*}} : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
//  // CHECK-MAX: memref.alloc() {{.*}} : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
//  // CHECK-MAX: d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>
//  // CHECK-AUTO-LABEL: func.func @matmul_auto_vs_max()
//  // CHECK-AUTO: d2m.view_layout {{.*}} -> memref<1x16x16x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
//  // CHECK-AUTO: d2m.view_layout {{.*}} -> memref<16x1x1x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
//  // CHECK-AUTO: d2m.generic {block_factors = [1, 1, 16], grid = #ttcore.grid<1x1>
//  // CHECK-AUTO: memref.alloc() {{.*}} : memref<16x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
//  // CHECK-AUTO: memref.alloc() {{.*}} : memref<1x16x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<65536x4096, 2>, #l1>
//  func.func @matmul_auto_vs_max() -> memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1> {
//    %lhs = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
//    %rhs = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
//    %out = memref.alloc() : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
//    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
//        ins(%lhs, %rhs : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>, memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>)
//        outs(%out : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>) {
//    ^compute0():
//      %0 = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
//      %1 = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
//      %2 = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
//      linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : memref<16x16x!ttcore.tile<32x32, f32>, #l1>, memref<16x16x!ttcore.tile<32x32, f32>, #l1>) outs(%2 : memref<16x16x!ttcore.tile<32x32, f32>, #l1>) {
//      ^bb0(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>, %out_elem: !ttcore.tile<32x32, f32>):
//        %9 = "d2m.tile_matmul"(%a, %b, %out_elem) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
//        linalg.yield %9 : !ttcore.tile<32x32, f32>
//      }
//    }
//    return %out : memref<1x1x16x16x!ttcore.tile<32x32, f32>, #ttcore.shard<65536x4096, 1>, #l1>
//  }
//
//  // CHECK-MAX-LABEL: func.func @single_reduction_non_matmul()
//  // CHECK-MAX: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>
//  // CHECK-AUTO-LABEL: func.func @single_reduction_non_matmul()
//  // CHECK-AUTO: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>
//  func.func @single_reduction_non_matmul() -> memref<1x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> {
//    %in = memref.alloc() : memref<1x1x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x32768, 1>, #l1>
//    %broadcast_in = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
//    %out = memref.alloc() : memref<1x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
//    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#reduceIn, #broadcast, #reduceOut], iterator_types = [#parallel, #reduction], threads = [#d2m.thread<compute>]}
//        ins(%in, %broadcast_in : memref<1x1x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x32768, 1>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
//        outs(%out : memref<1x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
//    ^compute0():
//    }
//    return %out : memref<1x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
//  }
//
//  // CHECK-AUTO-LABEL: func.func @auto_clamps_reduction_factor_at_four_tiles()
//  // CHECK-AUTO: d2m.view_layout {{.*}} -> memref<1x2x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
//  // CHECK-AUTO: d2m.view_layout {{.*}} -> memref<2x1x4x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
//  // CHECK-AUTO: d2m.generic {block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>
//  // CHECK-AUTO: memref.alloc() {{.*}} : memref<1x4x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<16384x4096, 2>, #l1>
//  // CHECK-AUTO: memref.alloc() {{.*}} : memref<4x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
//  func.func @auto_clamps_reduction_factor_at_four_tiles() -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> {
//    %lhs = memref.alloc() : memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>
//    %rhs = memref.alloc() : memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x32768, 1>, #l1>
//    %out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
//    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
//        ins(%lhs, %rhs : memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>, memref<1x1x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x32768, 1>, #l1>)
//        outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
//    ^compute0():
//      %0 = memref.alloc() : memref<1x8x!ttcore.tile<32x32, f32>, #l1>
//      %1 = memref.alloc() : memref<8x1x!ttcore.tile<32x32, f32>, #l1>
//      %2 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
//      linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = ["parallel", "parallel", "reduction"]} ins(%0, %1 : memref<1x8x!ttcore.tile<32x32, f32>, #l1>, memref<8x1x!ttcore.tile<32x32, f32>, #l1>) outs(%2 : memref<1x1x!ttcore.tile<32x32, f32>, #l1>) {
//      ^bb0(%a: !ttcore.tile<32x32, f32>, %b: !ttcore.tile<32x32, f32>, %out_elem: !ttcore.tile<32x32, f32>):
//        %9 = "d2m.tile_matmul"(%a, %b, %out_elem) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
//        linalg.yield %9 : !ttcore.tile<32x32, f32>
//      }
//    }
//    return %out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
//  }
//
//  // CHECK-AUTO-LABEL: func.func @multi_reduction_unchanged_under_auto()
//  // CHECK-AUTO: d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>
//  func.func @multi_reduction_unchanged_under_auto() -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> {
//    %in = memref.alloc() : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
//    %out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
//    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#multiReduceIn, #multiReduceOut], iterator_types = [#reduction, #reduction], threads = [#d2m.thread<compute>]}
//        ins(%in : memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>)
//        outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
//    ^compute0():
//    }
//    return %out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
//  }
//}
