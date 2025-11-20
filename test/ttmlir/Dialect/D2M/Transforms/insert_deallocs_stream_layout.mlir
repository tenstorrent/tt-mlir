// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-deallocs -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test that D2MInsertDeallocs correctly handles stream_layout operations
// and ensures that INPUT buffers are kept alive as long as the stream is used.

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

// CHECK-LABEL: func.func @stream_layout_input_kept_alive
func.func @stream_layout_input_kept_alive() -> memref<16x16x!ttcore.tile<32x32, f32>, #l1> {
  // Allocate DRAM buffer (INPUT to stream_layout)
  // CHECK: %[[INPUT:.*]] = memref.alloc(){{.*}}#dram>
  %input = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #dram>
  
  // Allocate L1 buffer (STORAGE for stream_layout)
  // CHECK: %[[STORAGE:.*]] = memref.alloc(){{.*}}#l1>
  %storage = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // Create stream_layout - result aliases both input and storage
  // CHECK: %[[STREAM:.*]] = "d2m.stream_layout"(%[[INPUT]], %[[STORAGE]])
  %stream = "d2m.stream_layout"(%input, %storage) : (memref<16x16x!ttcore.tile<32x32, f32>, #dram>, 
                                                      memref<16x16x!ttcore.tile<32x32, f32>, #l1>) 
                                                   -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // Allocate result buffer
  // CHECK: %[[RESULT:.*]] = memref.alloc(){{.*}}#l1>
  %result = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // Use stream in d2m.generic - INPUT must remain alive during this
  // CHECK: d2m.generic
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                affine_map<(d0, d1) -> (d0, d1)>],
               iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
               threads = [#d2m.thread<compute>]}
      ins(%stream : memref<16x16x!ttcore.tile<32x32, f32>, #l1>)
      outs(%result : memref<16x16x!ttcore.tile<32x32, f32>, #l1>) {
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
  
  // Deallocs should be placed AFTER d2m.generic completes
  // INPUT must not be deallocated before d2m.generic finishes
  // CHECK-NOT: memref.dealloc %[[INPUT]]
  // CHECK-NOT: memref.dealloc %[[STORAGE]]
  // CHECK: memref.dealloc %[[STORAGE]]
  // CHECK: memref.dealloc %[[INPUT]]
  
  return %result : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
}

// CHECK-LABEL: func.func @stream_layout_matmul_scenario
func.func @stream_layout_matmul_scenario() -> memref<16x16x!ttcore.tile<32x32, f32>, #l1> {
  // This test mimics the failing matmul scenario from the bug report
  
  // Allocate two input buffers in DRAM
  // CHECK: %[[INPUT_A:.*]] = memref.alloc(){{.*}}#dram>
  %input_a = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #dram>
  // CHECK: %[[INPUT_B:.*]] = memref.alloc(){{.*}}#dram>
  %input_b = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #dram>
  
  // Allocate L1 storage buffers for streaming
  // CHECK: %[[STORAGE_A:.*]] = memref.alloc(){{.*}}#l1>
  %storage_a = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: %[[STORAGE_B:.*]] = memref.alloc(){{.*}}#l1>
  %storage_b = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // Create stream_layout operations
  // CHECK: %[[STREAM_A:.*]] = "d2m.stream_layout"(%[[INPUT_A]], %[[STORAGE_A]])
  %stream_a = "d2m.stream_layout"(%input_a, %storage_a) : (memref<16x16x!ttcore.tile<32x32, f32>, #dram>, 
                                                            memref<16x16x!ttcore.tile<32x32, f32>, #l1>) 
                                                         -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  // CHECK: %[[STREAM_B:.*]] = "d2m.stream_layout"(%[[INPUT_B]], %[[STORAGE_B]])
  %stream_b = "d2m.stream_layout"(%input_b, %storage_b) : (memref<16x16x!ttcore.tile<32x32, f32>, #dram>, 
                                                            memref<16x16x!ttcore.tile<32x32, f32>, #l1>) 
                                                         -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // Deallocs MUST NOT appear here - streams are still used in d2m.generic
  // CHECK-NOT: memref.dealloc %[[INPUT_A]]
  // CHECK-NOT: memref.dealloc %[[INPUT_B]]
  // CHECK-NOT: memref.dealloc %[[STORAGE_A]]
  // CHECK-NOT: memref.dealloc %[[STORAGE_B]]
  
  // Allocate result buffer
  // CHECK: %[[RESULT:.*]] = memref.alloc(){{.*}}#l1>
  %result = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // Use streams in d2m.generic with tile_matmul
  // CHECK: d2m.generic
  d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                affine_map<(d0, d1, d2) -> (d2, d1)>,
                                affine_map<(d0, d1, d2) -> (d0, d1)>],
               iterator_types = [#ttcore.iterator_type<parallel>, 
                                #ttcore.iterator_type<parallel>, 
                                #ttcore.iterator_type<reduction>],
               threads = [#d2m.thread<compute>]}
      ins(%stream_a, %stream_b : memref<16x16x!ttcore.tile<32x32, f32>, #l1>,
                                 memref<16x16x!ttcore.tile<32x32, f32>, #l1>)
      outs(%result : memref<16x16x!ttcore.tile<32x32, f32>, #l1>) {
  ^compute0(%cb_a: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_b: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_out: !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>>):
    %0 = d2m.wait %cb_a : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    %1 = d2m.wait %cb_b : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    %2 = d2m.reserve %cb_out : !d2m.cb<memref<16x16x!ttcore.tile<32x32, f32>, #l1>> -> memref<16x16x!ttcore.tile<32x32, f32>, #l1>
    
    "d2m.tile_matmul_block"(%0, %1, %2) : (memref<16x16x!ttcore.tile<32x32, f32>, #l1>, 
                                           memref<16x16x!ttcore.tile<32x32, f32>, #l1>, 
                                           memref<16x16x!ttcore.tile<32x32, f32>, #l1>) -> ()
  }
  // CHECK: }
  
  // Now deallocs should appear AFTER d2m.generic completes
  // Storage buffers deallocated first, then input buffers
  // CHECK: memref.dealloc %[[STORAGE_B]]
  // CHECK: memref.dealloc %[[STORAGE_A]]
  // CHECK: memref.dealloc %[[INPUT_B]]
  // CHECK: memref.dealloc %[[INPUT_A]]
  
  return %result : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
}

// CHECK-LABEL: func.func @view_layout_aliasing
func.func @view_layout_aliasing() -> memref<8x8x!ttcore.tile<32x32, f32>, #l1> {
  // Test that view_layout aliasing is correctly handled
  
  // CHECK: %[[BUFFER:.*]] = memref.alloc(){{.*}}#l1>
  %buffer = memref.alloc() : memref<16x16x!ttcore.tile<32x32, f32>, #l1>
  
  // Create view into buffer
  // CHECK: %[[VIEW:.*]] = "d2m.view_layout"(%[[BUFFER]])
  %view = "d2m.view_layout"(%buffer) {shape = [8, 8]} : (memref<16x16x!ttcore.tile<32x32, f32>, #l1>) 
                                                       -> memref<8x8x!ttcore.tile<32x32, f32>, #l1>
  
  // Allocate result
  // CHECK: %[[RESULT:.*]] = memref.alloc(){{.*}}#l1>
  %result = memref.alloc() : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
  
  // Use view in operation
  // CHECK: d2m.generic
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                affine_map<(d0, d1) -> (d0, d1)>],
               iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
               threads = [#d2m.thread<compute>]}
      ins(%view : memref<8x8x!ttcore.tile<32x32, f32>, #l1>)
      outs(%result : memref<8x8x!ttcore.tile<32x32, f32>, #l1>) {
  ^compute0(%cb_in: !d2m.cb<memref<8x8x!ttcore.tile<32x32, f32>, #l1>>,
            %cb_out: !d2m.cb<memref<8x8x!ttcore.tile<32x32, f32>, #l1>>):
    %0 = d2m.wait %cb_in : !d2m.cb<memref<8x8x!ttcore.tile<32x32, f32>, #l1>> -> memref<8x8x!ttcore.tile<32x32, f32>, #l1>
    %1 = d2m.reserve %cb_out : !d2m.cb<memref<8x8x!ttcore.tile<32x32, f32>, #l1>> -> memref<8x8x!ttcore.tile<32x32, f32>, #l1>
    
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                     affine_map<(d0, d1) -> (d0, d1)>],
                    iterator_types = ["parallel", "parallel"]}
        ins(%0 : memref<8x8x!ttcore.tile<32x32, f32>, #l1>)
        outs(%1 : memref<8x8x!ttcore.tile<32x32, f32>, #l1>) {
    ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
      linalg.yield %arg0 : !ttcore.tile<32x32, f32>
    }
  }
  // CHECK: }
  
  // Buffer should only be deallocated once, not separately for buffer and view
  // CHECK: memref.dealloc %[[BUFFER]]
  // CHECK-NOT: memref.dealloc %[[VIEW]]
  
  return %result : memref<8x8x!ttcore.tile<32x32, f32>, #l1>
}
