// RUN: ttmlir-opt --d2m-insert-deallocs -o %t %s
// RUN: FileCheck %s --input-file=%t

// Test basic deallocation insertion for simple buffers

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

// CHECK-LABEL: func.func @simple_alloc_dealloc
func.func @simple_alloc_dealloc() {
  // CHECK: %[[BUF:.*]] = memref.alloc()
  %buf = memref.alloc() : memref<16x16xf32, #l1>
  
  // Use the buffer in a simple scf.for loop
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %val = memref.load %buf[%i, %j] : memref<16x16xf32, #l1>
      memref.store %val, %buf[%i, %j] : memref<16x16xf32, #l1>
    }
  }
  
  // CHECK: memref.dealloc %[[BUF]]
  return
}

// CHECK-LABEL: func.func @view_layout_aliasing
func.func @view_layout_aliasing() {
  // CHECK: %[[BUFFER:.*]] = memref.alloc()
  %buffer = memref.alloc() : memref<32x32xf32, #l1>
  
  // Create view into buffer (preserves element count: 32x32 = 1024 -> 16x64 = 1024)
  // CHECK: %[[VIEW:.*]] = d2m.view_layout %[[BUFFER]]
  %view = "d2m.view_layout"(%buffer) {shape = [16, 64]} : (memref<32x32xf32, #l1>) -> memref<16x64xf32, #l1>
  
  // Use view
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    %val = memref.load %view[%i, %c0] : memref<16x64xf32, #l1>
    memref.store %val, %view[%i, %c0] : memref<16x64xf32, #l1>
  }
  
  // Buffer should only be deallocated once, not separately for buffer and view
  // CHECK: memref.dealloc %[[BUFFER]]
  // CHECK-NOT: memref.dealloc %[[VIEW]]
  return
}

// CHECK-LABEL: func.func @stream_layout_aliasing
func.func @stream_layout_aliasing() {
  // CHECK: %[[INPUT:.*]] = memref.alloc(){{.*}}#dram>
  %input = memref.alloc() : memref<16x16xf32, #dram>
  
  // CHECK: %[[STORAGE:.*]] = memref.alloc(){{.*}}#l1>
  %storage = memref.alloc() : memref<16x16xf32, #l1>
  
  // Create stream_layout - result aliases both input and storage
  // CHECK: %[[STREAM:.*]] = "d2m.stream_layout"(%[[INPUT]], %[[STORAGE]])
  %stream = "d2m.stream_layout"(%input, %storage) : (memref<16x16xf32, #dram>, memref<16x16xf32, #l1>) -> memref<16x16xf32, #l1>
  
  // Use stream in nested loops - INPUT must remain alive during this
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c16 = arith.constant 16 : index
  scf.for %i = %c0 to %c16 step %c1 {
    scf.for %j = %c0 to %c16 step %c1 {
      %val = memref.load %stream[%i, %j] : memref<16x16xf32, #l1>
      memref.store %val, %stream[%i, %j] : memref<16x16xf32, #l1>
    }
  }
  
  // Deallocs should appear AFTER scf.for completes
  // INPUT must not be deallocated before scf.for finishes
  // CHECK: memref.dealloc %[[STORAGE]]
  // CHECK: memref.dealloc %[[INPUT]]
  return
}
