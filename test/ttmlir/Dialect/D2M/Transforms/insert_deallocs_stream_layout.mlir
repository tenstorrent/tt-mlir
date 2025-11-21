// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-deallocs -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that D2MInsertDeallocs correctly handles stream_layout operations
// and ensures that INPUT and STORAGE buffers are kept alive as long as the stream is used.

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>

// CHECK-LABEL: func.func @stream_layout_input_kept_alive
func.func @stream_layout_input_kept_alive() -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> {
  // Allocate DRAM buffer (INPUT to stream_layout)
  // CHECK: %[[INPUT:.*]] = memref.alloc(){{.*}}#l1>
  %input = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  
  // Allocate L1 buffer (STORAGE for stream_layout)
  // CHECK: %[[STORAGE:.*]] = memref.alloc(){{.*}}#l1>
  %storage = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  
  // Create stream_layout - result aliases both input and storage
  // CHECK: %[[STREAM:.*]] = "d2m.stream_layout"(%[[INPUT]], %[[STORAGE]])
  %stream = "d2m.stream_layout"(%input, %storage) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, 
                                                      memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) 
                                                   -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  
  // Allocate result buffer
  // CHECK: %[[RESULT:.*]] = memref.alloc(){{.*}}#l1>
  %result = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  
  // Use stream in d2m.generic - INPUT and STORAGE must remain alive during this
  // CHECK: d2m.generic
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [#map, #map],
               iterator_types = [#parallel, #parallel],
               threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
      ins(%stream : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>)
      outs(%result : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^datamovement0(%cb_in: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>,
                 %cb_out: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem_in = d2m.reserve %cb_in : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    %tx = d2m.dma %stream<#map>, %mem_in : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx
  }, {
  ^compute0(%cb_in: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_out: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    %0 = d2m.wait %cb_in : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    %1 = d2m.reserve %cb_out : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    
    affine.for %i = 0 to 2 {
      affine.for %j = 0 to 4 {
        %val = affine.load %0[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
        %exp = "d2m.tile_exp"(%val) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %exp, %1[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      }
    }
  }
  // CHECK: }
  
  // Deallocs should be placed AFTER d2m.generic completes
  // INPUT and STORAGE must not be deallocated before d2m.generic finishes
  // CHECK: memref.dealloc %[[STORAGE]]
  // CHECK: memref.dealloc %[[INPUT]]
  
  return %result : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
}

// CHECK-LABEL: func.func @multiple_streams
func.func @multiple_streams() -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> {
  // Test multiple stream_layout operations with shared buffers
  
  // CHECK: %[[INPUT_A:.*]] = memref.alloc(){{.*}}#l1>
  %input_a = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  // CHECK: %[[STORAGE_A:.*]] = memref.alloc(){{.*}}#l1>
  %storage_a = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  
  // CHECK: %[[INPUT_B:.*]] = memref.alloc(){{.*}}#l1>
  %input_b = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  // CHECK: %[[STORAGE_B:.*]] = memref.alloc(){{.*}}#l1>
  %storage_b = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  
  // CHECK: %[[STREAM_A:.*]] = "d2m.stream_layout"(%[[INPUT_A]], %[[STORAGE_A]])
  %stream_a = "d2m.stream_layout"(%input_a, %storage_a) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, 
                                                            memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) 
                                                         -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  // CHECK: %[[STREAM_B:.*]] = "d2m.stream_layout"(%[[INPUT_B]], %[[STORAGE_B]])
  %stream_b = "d2m.stream_layout"(%input_b, %storage_b) : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, 
                                                            memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) 
                                                         -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
  
  // Deallocs MUST NOT appear here - streams are still used in d2m.generic
  // CHECK-NOT: memref.dealloc %[[INPUT_A]]
  // CHECK-NOT: memref.dealloc %[[INPUT_B]]
  // CHECK-NOT: memref.dealloc %[[STORAGE_A]]
  // CHECK-NOT: memref.dealloc %[[STORAGE_B]]
  
  // CHECK: %[[RESULT:.*]] = memref.alloc(){{.*}}#l1>
  %result = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
  
  // CHECK: d2m.generic
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [#map, #map, #map],
               iterator_types = [#parallel, #parallel],
               threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
      ins(%stream_a, %stream_b : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>,
                                 memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>)
      outs(%result : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>) {
  ^datamovement0(%cb_a: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>,
                 %cb_b: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>,
                 %cb_out: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem_a = d2m.reserve %cb_a : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    %tx_a = d2m.dma %stream_a<#map>, %mem_a : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx_a
  }, {
  ^datamovement1(%cb_a: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>,
                 %cb_b: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>,
                 %cb_out: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
    %mem_b = d2m.reserve %cb_b : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
    %tx_b = d2m.dma %stream_b<#map>, %mem_b : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
    d2m.dma_wait %tx_b
  }, {
  ^datamovement2(%cb_a: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>,
                 %cb_b: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>,
                 %cb_out: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
  }, {
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
        %add = "d2m.tile_add"(%a, %b) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        affine.store %add, %2[%i, %j] : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      }
    }
  }
  // CHECK: }
  
  // All stream buffers should be deallocated AFTER d2m.generic completes
  // CHECK: memref.dealloc %[[STORAGE_B]]
  // CHECK: memref.dealloc %[[INPUT_B]]
  // CHECK: memref.dealloc %[[STORAGE_A]]
  // CHECK: memref.dealloc %[[INPUT_A]]
  
  return %result : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
}
