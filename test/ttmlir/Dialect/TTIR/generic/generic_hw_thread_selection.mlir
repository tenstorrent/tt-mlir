// RUN: ttmlir-opt --tt-register-device --ttir-generic-hw-thread-selection %s

#dram = #tt.memory_space<dram>
#l1_ = #tt.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #tt.iterator_type<parallel>
#reduction = #tt.iterator_type<reduction>

func.func @add(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    ttir.yield %cb0 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement1(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    ttir.yield %cb1 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
  }, {
  // CHECK-NOT: ^datamovement2
  // CHECK: ttir.yield [[CB2:%.*]] : {{.*}}
  // CHECK-NEXT: ttir.await [[CB2]] : {{.*}}
  ^datamovement2(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    ttir.await %cb2 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^compute(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x4x!tt.tile<32x32, f32>, #l1_>):
    ttir.await %cb0, %cb1 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<2x4x!tt.tile<32x32, f32>, #l1_>)
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 4 {
        %0 = affine.load %cb0[%arg2, %arg3] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %1 = affine.load %cb1[%arg2, %arg3] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
        %2 = "ttir.tile_add"(%0, %1) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
        affine.store %2, %cb2[%arg2, %arg3] : memref<2x4x!tt.tile<32x32, f32>, #l1_>
      }
    }
    ttir.yield %cb2 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>
}

func.func @matmul_single_core(%arg0: memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<1x1x4x2x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %arg1, %alloc) <{grid = #tt.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    ttir.yield %cb0 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^datamovement1(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    ttir.yield %cb1 : (memref<4x2x!tt.tile<32x32, f32>, #l1_>)
  }, {
  // CHECK-NOT: ^datamovement2
  // CHECK: ttir.yield [[CB2:%.*]] : {{.*}}
  // CHECK-NEXT: ttir.await [[CB2]] : {{.*}}
  ^datamovement2(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    ttir.await %cb2 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^compute(%cb0: memref<2x4x!tt.tile<32x32, f32>, #l1_>, %cb1: memref<4x2x!tt.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!tt.tile<32x32, f32>, #l1_>):
    ttir.await %cb0, %cb1 : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>)
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x4x!tt.tile<32x32, f32>, #l1_>, memref<4x2x!tt.tile<32x32, f32>, #l1_>, memref<2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
    ttir.yield %cb2 : (memref<2x2x!tt.tile<32x32, f32>, #l1_>)
  }) : (memref<1x1x2x4x!tt.tile<32x32, f32>, #l1_>, memref<1x1x4x2x!tt.tile<32x32, f32>, #l1_>, memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!tt.tile<32x32, f32>, #l1_>
}

func.func @tilize(%arg0: memref<2x4x128x192xf32, #l1_>) -> memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>
  "ttir.generic"(%arg0, %alloc) <{grid = #tt.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^datamovement0(%cb0: memref<128x192xf32, #l1_>, %cb1: memref<4x6x!tt.tile<32x32, f32>, #l1_>):
    ttir.yield %cb0 : (memref<128x192xf32, #l1_>)
  }, {
  // CHECK-NOT: ^datamovement1
  // CHECK: ttir.yield [[CB1:%.*]] : {{.*}}
  // CHECK-NEXT: ttir.await [[CB1]] : {{.*}}
  ^datamovement1(%cb0: memref<128x192xf32, #l1_>, %cb1: memref<4x6x!tt.tile<32x32, f32>, #l1_>):
    ttir.await %cb1 : (memref<4x6x!tt.tile<32x32, f32>, #l1_>)
  }, {
  ^compute(%cb0: memref<128x192xf32, #l1_>, %cb1: memref<4x6x!tt.tile<32x32, f32>, #l1_>):
    ttir.await %cb0 : (memref<128x192xf32, #l1_>)
    "ttir.tile_tilize_block"(%cb0, %cb1) : (memref<128x192xf32, #l1_>, memref<4x6x!tt.tile<32x32, f32>, #l1_>) -> ()
    ttir.yield %cb1 : (memref<4x6x!tt.tile<32x32, f32>, #l1_>)
  }) : (memref<2x4x128x192xf32, #l1_>, memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>) -> ()
  return %alloc : memref<2x4x4x6x!tt.tile<32x32, f32>, #l1_>
}
