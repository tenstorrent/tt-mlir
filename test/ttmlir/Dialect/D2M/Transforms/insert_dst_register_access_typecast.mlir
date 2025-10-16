// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-tile-compute-loops --d2m-insert-dst-register-access="max-dst-physical-size-tiles=32" --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that InsertDstRegisterAccess correctly inserts d2m.dst_reinterpret_cast operations
// when handling typecast operations with mismatched input/output types.

#l1_ = #ttcore.memory_space<l1>
#dst = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func private @typecast_f32_to_f16_generic
  func.func private @typecast_f32_to_f16_generic(
    %in0: memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #ttcore.memory_space<l1>>,
    %out0: memref<1x1x1x8x!ttcore.tile<32x32, f16>, #ttcore.shard<16384x2048>, #ttcore.memory_space<l1>>
  ) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #ttcore.memory_space<l1>>)
        outs(%out0 : memref<1x1x1x8x!ttcore.tile<32x32, f16>, #ttcore.shard<16384x2048>, #ttcore.memory_space<l1>>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>>, %arg1_cb: !d2m.cb<memref<1x8x!ttcore.tile<32x32, f16>, #ttcore.memory_space<l1>>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>> -> memref<1x8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x8x!ttcore.tile<32x32, f16>, #ttcore.memory_space<l1>>> -> memref<1x8x!ttcore.tile<32x32, f16>, #ttcore.memory_space<l1>>
      // CHECK: %[[DST:.*]] = d2m.acquire_dst() : memref<{{.*}}x!ttcore.tile<32x32, f32>, #dst>

      // CHECK: affine.for %{{.*}} = 0 to 1 {
      // CHECK-NEXT: affine.for %{{.*}} = 0 to 1 {
      // Copy from input CB to dst
      // CHECK: affine.load %{{.*}}[%{{.*}}, %{{.*}}] : memref<1x1x!ttcore.tile<32x32, f32>, strided<[8, 1], offset: ?>, #l1>
      // CHECK: affine.store %{{.*}}, %[[DST]]

      // CHECK: affine.for %{{.*}} = 0 to 1 {
      // CHECK-NEXT: affine.for %{{.*}} = 0 to 1 {
      // Typecast with dst_reinterpret_cast to dst type
      // CHECK: %[[DST_LOAD:.*]] = affine.load %[[DST]]
      // CHECK: %[[TYPECAST:.*]] = "d2m.tile_typecast"(%[[DST_LOAD]]) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f16>
      // CHECK: %[[CAST_TO_DST:.*]] = "d2m.dst_reinterpret_cast"(%[[TYPECAST]]) : (!ttcore.tile<32x32, f16>) -> !ttcore.tile<32x32, f32>
      // CHECK: affine.store %[[CAST_TO_DST]], %[[DST]]

      // CHECK: affine.for %{{.*}} = 0 to 1 {
      // CHECK-NEXT: affine.for %{{.*}} = 0 to 1 {
      // Copy from dst to output CB with dst_reinterpret_cast for type conversion
      // CHECK: %[[DST_LOAD2:.*]] = affine.load %[[DST]]
      // CHECK: %[[CAST_TO_OUTPUT:.*]] = "d2m.dst_reinterpret_cast"(%[[DST_LOAD2]]) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f16>
      // CHECK: affine.store %[[CAST_TO_OUTPUT]], %{{.*}}[%{{.*}}, %{{.*}}] : memref<1x1x!ttcore.tile<32x32, f16>, strided<[8, 1], offset: ?>, #l1>

      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %arg2 = %c0 to %c1 step %c1 {
        scf.for %arg3 = %c0 to %c8 step %c8 {
          %subview = memref.subview %cb0[%arg2, %arg3] [1, 8] [1, 1] : memref<1x8x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> to memref<1x8x!ttcore.tile<32x32, f32>, strided<[8, 1], offset: ?>, #ttcore.memory_space<l1>>
          %subview_0 = memref.subview %cb1[%arg2, %arg3] [1, 8] [1, 1] : memref<1x8x!ttcore.tile<32x32, f16>, #ttcore.memory_space<l1>> to memref<1x8x!ttcore.tile<32x32, f16>, strided<[8, 1], offset: ?>, #ttcore.memory_space<l1>>
          linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]} ins(%subview : memref<1x8x!ttcore.tile<32x32, f32>, strided<[8, 1], offset: ?>, #ttcore.memory_space<l1>>) outs(%subview_0 : memref<1x8x!ttcore.tile<32x32, f16>, strided<[8, 1], offset: ?>, #ttcore.memory_space<l1>>) {
          ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f16>):
            %0 = "d2m.tile_typecast"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f16>
            linalg.yield %0 : !ttcore.tile<32x32, f16>
          }
        }
      }
    }
    return
  }
}
