// RUN: ttmlir-opt --convert-ttmetal-to-emitc %s | FileCheck %s

module {
  func.func @buffer_operations() {
    %system_mem = memref.alloc() : memref<64x128xf32, #ttcore.memory_space<system>>
    %device_mem = memref.alloc() : memref<64x128xf32, #ttcore.memory_space<dram>>
    
    // Test write buffer operation: input=system, output=device
    // CHECK: emitc.call_opaque "tt::tt_metal::enqueue_write_buffer"
    "ttmetal.enqueue_write_buffer"(%system_mem, %device_mem) : (memref<64x128xf32, #ttcore.memory_space<system>>, memref<64x128xf32, #ttcore.memory_space<dram>>) -> ()
    
    // Test read buffer operation: input=device, output=system
    // CHECK: emitc.call_opaque "tt::tt_metal::enqueue_read_buffer" 
    "ttmetal.enqueue_read_buffer"(%device_mem, %system_mem) : (memref<64x128xf32, #ttcore.memory_space<dram>>, memref<64x128xf32, #ttcore.memory_space<system>>) -> ()
    
    memref.dealloc %system_mem : memref<64x128xf32, #ttcore.memory_space<system>>
    memref.dealloc %device_mem : memref<64x128xf32, #ttcore.memory_space<dram>>
    
    return
  }
}