// RUN: ttmlir-opt --convert-ttmetal-to-emitc %s | FileCheck %s

module {
  func.func @simple_enqueue_program() {
    // Create some test buffers with device memory space
    %buf1 = memref.alloc() : memref<32x32xf32, #ttcore.memory_space<dram>>
    %buf2 = memref.alloc() : memref<32x32xf32, #ttcore.memory_space<dram>>
    
    // Test basic enqueue program operation with minimal attributes
    // CHECK: emitc.call_opaque "tt::tt_metal::enqueue_program"
    "ttmetal.enqueue_program"(%buf1, %buf2) <{
      cb_ports = array<i64: 0, 1>,
      kernelConfigs = [],
      operandSegmentSizes = array<i32: 2, 0>
    }> : (memref<32x32xf32, #ttcore.memory_space<dram>>, memref<32x32xf32, #ttcore.memory_space<dram>>) -> ()
    
    // Clean up buffers
    memref.dealloc %buf1 : memref<32x32xf32, #ttcore.memory_space<dram>>
    memref.dealloc %buf2 : memref<32x32xf32, #ttcore.memory_space<dram>>
    
    return
  }
}