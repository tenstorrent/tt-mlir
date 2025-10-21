// RUN: ttmlir-opt --ttcore-register-device --d2m-generic-hw-thread-selection -o %t %s
// RUN: FileCheck %s --input-file=%t

#dram = #ttcore.memory_space<dram>
#l1_ = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

// CHECK: func.func @add
func.func @add(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, %arg1: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  "d2m.generic"(%arg0, %arg1, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1:  !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2:  !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
  }, {
  ^datamovement1(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1:  !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2:  !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
  }, {
  // CHECK-NOT: ^datamovement2
  // CHECK: ^compute
  ^datamovement2(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1:  !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2:  !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
  }, {
  ^compute(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1:  !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb2:  !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %mem0 = d2m.pop %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %mem1 = d2m.pop %cb1 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    affine.for %arg2 = 0 to 2 {
      affine.for %arg3 = 0 to 4 {
        %0 = affine.load %mem0[%arg2, %arg3] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %1 = affine.load %mem1[%arg2, %arg3] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
        %2 = "d2m.tile_add"(%0, %1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %2, %mem2[%arg2, %arg3] : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      }
    }
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
}

// CHECK: func.func @matmul_single_core
func.func @matmul_single_core(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, %arg1: memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
  "d2m.generic"(%arg0, %arg1, %alloc) <{block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>], operandSegmentSizes = array<i32: 2, 1>}> ({
  ^datamovement0(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1:  !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2:  !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
  }, {
  ^datamovement1(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1:  !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2:  !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
  }, {
  // CHECK-NOT: ^datamovement2
  // CHECK: ^compute
  ^datamovement2(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1:  !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2:  !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
  }, {
  ^compute(%cb0: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>, %cb1:  !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>>, %cb2:  !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>>):
    %mem0 = d2m.pop %cb0 : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %mem1 = d2m.pop %cb1 : !d2m.cb<memref<4x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x2x!ttcore.tile<32x32, f32>, #l1_>
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_matmul_block"(%mem0, %mem1, %mem2) : (memref<2x4x!ttcore.tile<32x32, f32>, #l1_>, memref<4x2x!ttcore.tile<32x32, f32>, #l1_>, memref<2x2x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>, memref<1x1x4x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>, memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>) -> ()
  return %alloc : memref<1x1x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096>, #l1_>
}

// CHECK: func.func @tilize
func.func @tilize(%arg0: memref<2x4x128x192xf32, #ttcore.shard<768x4>, #l1_>) -> memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_> {
  %alloc = memref.alloc() {alignment = 64 : i64} : memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>
  "d2m.generic"(%arg0, %alloc) <{block_factors = [1, 1], grid = #ttcore.grid<2x4>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>], operandSegmentSizes = array<i32: 1, 1>}> ({
  ^datamovement0(%cb0: !d2m.cb<memref<128x192xf32, #l1_>>, %cb1:  !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>):
  }, {
  // CHECK-NOT: ^datamovement2
  ^datamovement1(%cb0: !d2m.cb<memref<128x192xf32, #l1_>>, %cb1:  !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>):
  }, {
  ^compute(%cb0: !d2m.cb<memref<128x192xf32, #l1_>>, %cb1:  !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>>):
    %mem0 = d2m.pop %cb0 : !d2m.cb<memref<128x192xf32, #l1_>> -> memref<128x192xf32, #l1_>
    %mem1 = d2m.reserve %cb1 : !d2m.cb<memref<4x6x!ttcore.tile<32x32, f32>, #l1_>> -> memref<4x6x!ttcore.tile<32x32, f32>, #l1_>
    "d2m.tile_tilize_block"(%mem0, %mem1) : (memref<128x192xf32, #l1_>, memref<4x6x!ttcore.tile<32x32, f32>, #l1_>) -> ()
  }) : (memref<2x4x128x192xf32, #ttcore.shard<768x4>, #l1_>, memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>) -> ()
  return %alloc : memref<2x4x4x6x!ttcore.tile<32x32, f32>, #ttcore.shard<24576x4096>, #l1_>
}

#l1 = #ttcore.memory_space<l1>
// CHECK: func.func @mergeNonTrivialDatamovementThreads
func.func @mergeNonTrivialDatamovementThreads(%arg0: memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) {
  %alloc = memref.alloc() {address = 296352 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_0 = memref.alloc() {address = 361888 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream = "d2m.stream_layout"(%alloc, %alloc_0) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  %alloc_1 = memref.alloc() {address = 427424 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_2 = memref.alloc() {address = 492960 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream_3 = "d2m.stream_layout"(%alloc_1, %alloc_2) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  %alloc_4 = memref.alloc() {address = 558496 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_5 = memref.alloc() {address = 624032 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream_6 = "d2m.stream_layout"(%alloc_4, %alloc_5) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
      ins(%stream, %stream_3 : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>)
      outs(%stream_6 : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>)  {
  ^datamovement0(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem0 = d2m.reserve %cb0 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %stream<#map>, %mem0 : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  ^datamovement1(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem1 = d2m.reserve %cb1 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %stream_3<#map>, %mem1 : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  // CHECK: ^datamovement0
  // CHECK: d2m.dma [[STREAM0:%.*]]<#map>, [[CB0:%.*]]
  // CHECK: ^datamovement1
  // CHECK: d2m.dma [[STREAM1:%.*]]<#map>, [[CB1:%.*]]
  // CHECK-NOT: ^datamovement2
  // CHECK: ^compute
  ^datamovement2(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem2 = d2m.pop %cb2 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %mem2, %stream_6<#map> : (memref<4x4x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  ^compute0(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
  }
  return
}


// CHECK: func.func @mergeManyNonTrivialDatamovementThreads
func.func @mergeManyNonTrivialDatamovementThreads(%arg0: memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) {
  %alloc = memref.alloc() {address = 296352 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_0 = memref.alloc() {address = 361888 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream = "d2m.stream_layout"(%alloc, %alloc_0) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  %alloc_1 = memref.alloc() {address = 427424 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_2 = memref.alloc() {address = 492960 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream_3 = "d2m.stream_layout"(%alloc_1, %alloc_2) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  %alloc_4 = memref.alloc() {address = 558496 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_5 = memref.alloc() {address = 624032 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream_6 = "d2m.stream_layout"(%alloc_4, %alloc_5) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>

  %alloc_8 = memref.alloc() {address = 296352 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_9 = memref.alloc() {address = 361888 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream_9 = "d2m.stream_layout"(%alloc, %alloc_0) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>

  d2m.generic {block_factors = [1, 1],
        grid = #ttcore.grid<1x1>,
        indexing_maps = [#map, #map, #map, #map],
        iterator_types = [#parallel, #parallel],
        threads = [
            #d2m.thread<datamovement>,
            #d2m.thread<datamovement>,
            #d2m.thread<datamovement>,
            #d2m.thread<datamovement>,
            #d2m.thread<compute>]}
      ins(%stream, %stream_3, %stream_9 :
        memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>,
        memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>,
        memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
        )
      outs(%stream_6 : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>)  {
  // CHECK: ^datamovement0
  // CHECK: d2m.dma [[STREAM0:%.*]]<#map>, [[CB0:%.*]]
  // CHECK: d2m.dma_wait
  // CHECK: d2m.dma [[STREAM2:%.*]]<#map>, [[CB2:%.*]]
  // CHECK: d2m.dma_wait
  ^datamovement0(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem0 = d2m.reserve %cb0 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %stream<#map>, %mem0 : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  // CHECK: ^datamovement1
  // CHECK: d2m.dma [[STREAM1:%.*]]<#map>, [[CB1:%.*]]
  // CHECK: d2m.dma_wait
  // CHECK: d2m.dma [[CB3:%.*]], [[STREAM3:%.*]]<#map>
  // CHECK: d2m.dma_wait
  ^datamovement1(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem1 = d2m.reserve %cb1 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %stream_3<#map>, %mem1 : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  ^datamovement2(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %stream_9<#map>, %mem2 : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  // CHECK-NOT: ^datamovement2
  // CHECK: ^compute
  ^datamovement3(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem3 = d2m.pop %cb3 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %mem3, %stream_6<#map> : (memref<4x4x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  ^compute0(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
  }
  return
}


// CHECK: func.func @mergeManyNonTrivialDatamovementThreadsNoCompute
func.func @mergeManyNonTrivialDatamovementThreadsNoCompute(%arg0: memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) {
  %alloc = memref.alloc() {address = 296352 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_0 = memref.alloc() {address = 361888 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream = "d2m.stream_layout"(%alloc, %alloc_0) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  %alloc_1 = memref.alloc() {address = 427424 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_2 = memref.alloc() {address = 492960 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream_3 = "d2m.stream_layout"(%alloc_1, %alloc_2) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  %alloc_4 = memref.alloc() {address = 558496 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_5 = memref.alloc() {address = 624032 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream_6 = "d2m.stream_layout"(%alloc_4, %alloc_5) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>

  %alloc_8 = memref.alloc() {address = 296352 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_9 = memref.alloc() {address = 361888 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream_9 = "d2m.stream_layout"(%alloc, %alloc_0) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>

  d2m.generic {block_factors = [1, 1],
        grid = #ttcore.grid<1x1>,
        indexing_maps = [#map, #map, #map, #map],
        iterator_types = [#parallel, #parallel],
        threads = [
            #d2m.thread<datamovement>,
            #d2m.thread<datamovement>,
            #d2m.thread<datamovement>,
            #d2m.thread<datamovement>
            ]}
      ins(%stream, %stream_3, %stream_9 :
        memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>,
        memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>,
        memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
        )
      outs(%stream_6 : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>)  {
  // CHECK: ^datamovement0
  // CHECK: d2m.dma [[STREAM0:%.*]]<#map>, [[CB0:%.*]]
  // CHECK: d2m.dma_wait
  // CHECK: d2m.dma [[STREAM2:%.*]]<#map>, [[CB2:%.*]]
  // CHECK: d2m.dma_wait
  ^datamovement0(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem0 = d2m.reserve %cb0 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %stream<#map>, %mem0 : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  // CHECK: ^datamovement1
  // CHECK: d2m.dma [[STREAM1:%.*]]<#map>, [[CB1:%.*]]
  // CHECK: d2m.dma_wait
  // CHECK: d2m.dma [[CB3:%.*]], [[STREAM3:%.*]]<#map>
  // CHECK: d2m.dma_wait
  ^datamovement1(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem1 = d2m.reserve %cb1 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %stream_3<#map>, %mem1 : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  ^datamovement2(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %stream_9<#map>, %mem2 : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  // CHECK-NOT: ^datamovement2
  // CHECK-NOT: ^compute
  ^datamovement3(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem3 = d2m.pop %cb3 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %mem3, %stream_6<#map> : (memref<4x4x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }
  return
}


// CHECK: func.func @dontMergePreFusedDMThreads
func.func @dontMergePreFusedDMThreads(%arg0: memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) {
  %alloc = memref.alloc() {address = 296352 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_0 = memref.alloc() {address = 361888 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream = "d2m.stream_layout"(%alloc, %alloc_0) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  %alloc_1 = memref.alloc() {address = 427424 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_2 = memref.alloc() {address = 492960 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream_3 = "d2m.stream_layout"(%alloc_1, %alloc_2) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  %alloc_4 = memref.alloc() {address = 558496 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_5 = memref.alloc() {address = 624032 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream_6 = "d2m.stream_layout"(%alloc_4, %alloc_5) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>

  %alloc_8 = memref.alloc() {address = 296352 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %alloc_9 = memref.alloc() {address = 361888 : i64, alignment = 16 : i64} : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>
  %stream_9 = "d2m.stream_layout"(%alloc, %alloc_0) : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>, memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1>) -> memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>

  d2m.generic {block_factors = [1, 1],
        grid = #ttcore.grid<1x1>,
        indexing_maps = [#map, #map, #map, #map],
        iterator_types = [#parallel, #parallel],
        threads = [
            #d2m.thread<datamovement>,
            #d2m.thread<datamovement>,
            #d2m.thread<compute>
            ]}
      ins(%stream, %stream_3, %stream_9 :
        memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>,
        memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>,
        memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
        )
      outs(%stream_6 : memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>)  {
  // CHECK: ^datamovement0
  ^datamovement0(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem0 = d2m.reserve %cb0 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %stream<#map>, %mem0 : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  // CHECK: ^datamovement1
  ^datamovement1(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem1 = d2m.reserve %cb1 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %stream_3<#map>, %mem1 : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
    %mem2 = d2m.reserve %cb2 : !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<4x4x!ttcore.tile<32x32, f32>, #l1>
    %tx2 = d2m.dma %stream_9<#map>, %mem2 : (memref<1x1x4x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<4x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx2
  }, {
  // CHECK: ^compute
  ^compute0(%cb0: !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb1:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>, %cb3:  !d2m.cb<memref<4x4x!ttcore.tile<32x32, f32>, #l1>>):
  }
  return
}
