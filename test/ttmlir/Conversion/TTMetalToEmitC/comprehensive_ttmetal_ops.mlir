// RUN: ttmlir-opt --convert-ttmetal-to-emitc %s | FileCheck %s

module {
  func.func @comprehensive_ttmetal_operations() {
    // Test buffer allocation with different memory spaces
    // CHECK: %{{[0-9]+}} = emitc.call_opaque "util_create_buffer"{{.*}} -> !emitc.ptr<!emitc.opaque<"::tt::tt_metal::Buffer">>
    %system_buf = memref.alloc() : memref<128x256xf32, #ttcore.memory_space<system>>
    
    // CHECK: %{{[0-9]+}} = emitc.call_opaque "util_create_buffer"{{.*}} -> !emitc.ptr<!emitc.opaque<"::tt::tt_metal::Buffer">>
    %device_buf = memref.alloc() : memref<128x256xf32, #ttcore.memory_space<dram>>
    
    // CHECK: %{{[0-9]+}} = emitc.call_opaque "util_create_buffer"{{.*}} -> !emitc.ptr<!emitc.opaque<"::tt::tt_metal::Buffer">>
    %device_buf2 = memref.alloc() : memref<32x32xf32, #ttcore.memory_space<dram>>

    // Test enqueue write buffer operation (host to device transfer)
    // CHECK: emitc.call_opaque "tt::tt_metal::enqueue_write_buffer"
    "ttmetal.enqueue_write_buffer"(%system_buf, %device_buf) : (memref<128x256xf32, #ttcore.memory_space<system>>, memref<128x256xf32, #ttcore.memory_space<dram>>) -> ()

    // Test enqueue program operation with empty kernel configs
    // CHECK: emitc.call_opaque "tt::tt_metal::enqueue_program"
    "ttmetal.enqueue_program"(%device_buf, %device_buf2) <{
      cb_ports = array<i64: 0, 1>,
      kernelConfigs = [],
      operandSegmentSizes = array<i32: 2, 0>
    }> : (memref<128x256xf32, #ttcore.memory_space<dram>>, memref<32x32xf32, #ttcore.memory_space<dram>>) -> ()

    // Test enqueue read buffer operation (device to host transfer)
    // CHECK: emitc.call_opaque "tt::tt_metal::enqueue_read_buffer"
    "ttmetal.enqueue_read_buffer"(%device_buf, %system_buf) : (memref<128x256xf32, #ttcore.memory_space<dram>>, memref<128x256xf32, #ttcore.memory_space<system>>) -> ()

    // Test buffer deallocation
    // CHECK: emitc.call_opaque "util_deallocate_buffer"
    memref.dealloc %system_buf : memref<128x256xf32, #ttcore.memory_space<system>>
    
    // CHECK: emitc.call_opaque "util_deallocate_buffer"
    memref.dealloc %device_buf : memref<128x256xf32, #ttcore.memory_space<dram>>
    
    // CHECK: emitc.call_opaque "util_deallocate_buffer"
    memref.dealloc %device_buf2 : memref<32x32xf32, #ttcore.memory_space<dram>>

    return
  }
}