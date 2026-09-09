// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @host_scalar_load_store
  // CHECK: memref.load
  // CHECK: memref.store
  // CHECK-NOT: ttkernel.pack_tile
  func.func @host_scalar_load_store(%arg0: memref<1x1xbf16>) -> memref<bf16> attributes {tt.function_type = "forward_device"} {
    %c0 = arith.constant 0 : index
    %0 = memref.load %arg0[%c0, %c0] : memref<1x1xbf16>
    %alloc = memref.alloc() : memref<bf16>
    memref.store %0, %alloc[] : memref<bf16>
    return %alloc : memref<bf16>
  }
}
