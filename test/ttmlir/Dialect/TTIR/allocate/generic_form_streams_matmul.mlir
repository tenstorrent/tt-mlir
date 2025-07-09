// RUN: ttmlir-opt --ttcore-register-device --ttir-allocate --canonicalize %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul_single_core_stream(%arg0: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, %arg1: memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  // CHECK-NOT: "ttir.view_layout"
  // CHECK: [[lhs:%[a-z0-9_]+]] = "ttir.stream_layout"(%arg0,
  %0 = "ttir.view_layout"(%arg0) <{viewMap = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  // CHECK-NOT: "ttir.view_layout"
  // CHECK: [[rhs:%[a-z0-9_]+]] = "ttir.stream_layout"(%arg1,
  %1 = "ttir.view_layout"(%arg1) <{viewMap = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  // CHECK: ttir.generic{{.*}}
  // CHECK-NEXT: ins([[lhs]], [[rhs]] : {{.*}})
  // CHECK-NEXT: outs([[out:%[a-z0-9_]+]] : {{.*}})
  "ttir.generic"(%0, %1, %alloc) <{block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%cb0: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<2x2x!ttcore.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

func.func @matmul_multi_core(%arg0: memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, %arg1: memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  // CHECK-NOT: "ttir.view_layout"
  // CHECK: [[lhs:%[a-z0-9_]+]] = "ttir.stream_layout"(%arg0,
  %0 = "ttir.view_layout"(%arg0) <{viewMap = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  // CHECK-NOT: "ttir.view_layout"
  // CHECK: [[rhs:%[a-z0-9_]+]] = "ttir.stream_layout"(%arg1,
  %1 = "ttir.view_layout"(%arg1) <{viewMap = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> : (memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  // CHECK: ttir.generic{{.*}}
  // CHECK-NEXT: ins([[lhs]], [[rhs]] : {{.*}})
  // CHECK-NEXT: outs([[out:%[a-z0-9_]+]] : {{.*}})
  "ttir.generic"(%0, %1, %alloc) <{block_factors = [1, 1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<4x8x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
}

func.func @matmul_multi_core_transpose(%arg0: memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, %arg1: memref<4x4x8x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  // CHECK-NOT: "ttir.view_layout"
  // CHECK: [[lhs:%[a-z0-9_]+]] = "ttir.stream_layout"(%arg0,
  %0 = "ttir.view_layout"(%arg0) <{viewMap = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>}> : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  // CHECK-NOT: "ttir.view_layout"
  // CHECK: [[rhs:%[a-z0-9_]+]] = "ttir.stream_layout"(%arg1, {{.*}}, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>,
  %1 = "ttir.view_layout"(%arg1) <{viewMap = affine_map<(d0, d1, d2, d3) -> (d0, d1, d3, d2)>}> : (memref<4x4x8x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>, #l1_>
  // CHECK: ttir.generic{{.*}}
  // CHECK-NEXT: ins([[lhs]], [[rhs]] : {{.*}})
  // CHECK-NEXT: outs([[out:%[a-z0-9_]+]] : {{.*}})
  "ttir.generic"(%0, %1, %alloc) <{block_factors = [1, 1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#ttir.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%cb0: memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, %cb1: memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, %cb2: memref<4x8x!ttcore.tile<32x32, f32>, #l1_>):
    "ttir.tile_matmul_block"(%cb0, %cb1, %cb2) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<4x8x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>, #l1_>, memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
}
