// RUN: ttmlir-opt --ttcore-register-device --d2m-allocate --canonicalize -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>

func.func @matmul_single_core_stream(%arg0: memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, %arg1: memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  // CHECK-NOT: "d2m.view_layout"
  // CHECK: [[lhs:%[a-z0-9_]+]] = "d2m.stream_layout"(%arg0,
  %0 = "d2m.view_layout"(%arg0) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  // CHECK-NOT: "d2m.view_layout"
  // CHECK: [[rhs:%[a-z0-9_]+]] = "d2m.stream_layout"(%arg1,
  %1 = "d2m.view_layout"(%arg1) : (memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  // CHECK: d2m.generic{{.*}}
  // CHECK-NEXT: ins([[lhs]], [[rhs]] : {{.*}})
  // CHECK-NEXT: outs([[out:%[a-z0-9_]+]] : {{.*}})
  "d2m.generic"(%0, %1, %alloc) <{block_factors = [1, 1, 2], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%cb0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    %lhs = d2m.wait %cb0 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    %rhs = d2m.wait %cb1 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    %out = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_matmul_block"(%lhs, %rhs, %out) : (memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

func.func @matmul_multi_core(%arg0: memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, %arg1: memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  // CHECK-NOT: "d2m.view_layout"
  // CHECK: [[lhs:%[a-z0-9_]+]] = "d2m.stream_layout"(%arg0,
  %0 = "d2m.view_layout"(%arg0) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  // CHECK-NOT: "d2m.view_layout"
  // CHECK: [[rhs:%[a-z0-9_]+]] = "d2m.stream_layout"(%arg1,
  %1 = "d2m.view_layout"(%arg1) : (memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  // CHECK: d2m.generic{{.*}}
  // CHECK-NEXT: ins([[lhs]], [[rhs]] : {{.*}})
  // CHECK-NEXT: outs([[out:%[a-z0-9_]+]] : {{.*}})
  "d2m.generic"(%0, %1, %alloc) <{block_factors = [1, 1, 4], grid = #ttcore.grid<2x4>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%cb0: !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>>):
    %lhs = d2m.wait %cb0 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
    %rhs = d2m.wait %cb1 : !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>> -> memref<6x8x!ttcore.tile<32x32, f32>, #l1_>
    %out = d2m.reserve %cb2 : !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x8x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_matmul_block"(%lhs, %rhs, %out) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<4x8x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
}

func.func @matmul_multi_core_transpose(%arg0: memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>, %arg1: memref<4x4x8x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
  // CHECK-NOT: "d2m.view_layout"
  // CHECK: [[lhs:%[a-z0-9_]+]] = "d2m.stream_layout"(%arg0,
  %0 = "d2m.view_layout"(%arg0) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
  // CHECK-NOT: "d2m.view_layout"
  // CHECK: [[rhs:%[a-z0-9_]+]] = "d2m.stream_layout"(%arg1, {{.*}}, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>,
  %1 = "d2m.view_layout"(%arg1) : (memref<4x4x8x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>, #l1_>
  // CHECK: d2m.generic{{.*}}
  // CHECK-NEXT: ins([[lhs]], [[rhs]] : {{.*}})
  // CHECK-NEXT: outs([[out:%[a-z0-9_]+]] : {{.*}})
  "d2m.generic"(%0, %1, %alloc) <{block_factors = [1, 1, 4], grid = #ttcore.grid<2x4>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^bb0(%cb0: !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>, %cb1: !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>>, %cb2: !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>>):
    %lhs = d2m.wait %cb0 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
    %rhs = d2m.wait %cb1 : !d2m.cb<memref<6x8x!ttcore.tile<32x32, f32>, #l1_>> -> memref<6x8x!ttcore.tile<32x32, f32>, #l1_>
    %out = d2m.reserve %cb2 : !d2m.cb<memref<4x8x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x8x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_matmul_block"(%lhs, %rhs, %out) : (memref<4x6x!ttcore.tile<32x32, f32>, #l1_>, memref<6x8x!ttcore.tile<32x32, f32>, #l1_>, memref<4x8x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<4x4x6x8x!ttcore.tile<32x32, f32>, #ttcore.view<(d0, d1, d2, d3) -> (d1, d0, d3, d2)>, #l1_>, memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>) -> ()
  return %alloc : memref<2x4x4x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096>, #l1_>
}
