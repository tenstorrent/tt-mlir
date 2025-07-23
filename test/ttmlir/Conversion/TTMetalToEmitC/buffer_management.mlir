// RUN: ttmlir-opt --convert-ttmetal-to-emitc %s | FileCheck %s

module {
  func.func @buffer_management() {
    // Test buffer allocation
    // CHECK: %{{[0-9]+}} = emitc.call_opaque "util_create_buffer"{{.*}} -> !emitc.ptr<!emitc.opaque<"::tt::tt_metal::Buffer">>
    %buf1 = memref.alloc() : memref<32x32xf32>
    
    // Test buffer deallocation  
    // CHECK: emitc.call_opaque "util_deallocate_buffer"
    memref.dealloc %buf1 : memref<32x32xf32>
    
    return
  }
}