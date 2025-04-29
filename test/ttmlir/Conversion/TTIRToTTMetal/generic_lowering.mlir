// RUN: ttmlir-opt --tt-register-device --convert-ttir-to-ttmetal %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #tt.memory_space<l1>
module {
  func.func @generic0(%arg0: memref<1x1x8x24x!tt.tile<32x32, f32>, #tt.shard<98304x4096>, #l1_>, %arg1: memref<1x1x24x32x!tt.tile<32x32, f32>, #tt.shard<131072x4096>, #l1_>) -> memref<1x1x8x32x!tt.tile<32x32, f32>, #tt.shard<131072x4096>, #l1_> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8x1x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>
    %alloc_0 = memref.alloc() {alignment = 64 : i64, address = 0x10000} : memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.shard<12288x4096>, #l1_>
    %stream = "ttir.stream_layout"(%arg0, %alloc_0) : (memref<1x1x8x24x!tt.tile<32x32, f32>, #tt.shard<98304x4096>, #l1_>, memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.shard<12288x4096>, #l1_>) -> memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.view<map(4)>, #l1_>
    %alloc_1 = memref.alloc() {alignment = 64 : i64, address = 0x15000} : memref<8x8x3x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>
    %stream_2 = "ttir.stream_layout"(%arg1, %alloc_1) : (memref<1x1x24x32x!tt.tile<32x32, f32>, #tt.shard<131072x4096>, #l1_>, memref<8x8x3x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>) -> memref<8x8x3x4x!tt.tile<32x32, f32>, #tt.view<map(4)>, #l1_>
    // CHECK: "ttmetal.enqueue_program"
    // CHECK-SAME: {{.*}}core_ranges = [#ttmetal.core_range<0x0, 8x8>, #ttmetal.core_range<0x0, 8x8>, #ttmetal.core_range<0x0, 8x8>]
    // CHECK-SAME: {{.*}}threads = [#ttir.thread<datamovement, @datamovement_kernel0>, #ttir.thread<datamovement, @datamovement_kernel1>, #ttir.thread<compute, @compute_kernel2>]
    ttir.generic {grid = #tt.grid<8x8>, indexing_maps = [], iterator_types = [], threads = [#ttir.thread<datamovement, @datamovement_kernel0>, #ttir.thread<datamovement, @datamovement_kernel1>, #ttir.thread<compute, @compute_kernel2>]}
        ins(%stream, %stream_2 : memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.view<map(4)>, #l1_>, memref<8x8x3x4x!tt.tile<32x32, f32>, #tt.view<map(4)>, #l1_>)
        outs(%alloc : memref<8x8x1x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>)
    %view = "ttir.view_layout"(%alloc) : (memref<8x8x1x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>) -> memref<1x1x8x32x!tt.tile<32x32, f32>, #tt.shard<131072x4096>, #l1_>
    return %view : memref<1x1x8x32x!tt.tile<32x32, f32>, #tt.shard<131072x4096>, #l1_>
  }

  func.func private @datamovement_kernel0(%arg0: memref<1x3x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!tt.tile<32x32, f32>, #l1_>, %arg3: !ttir.semaphore, %arg4: !ttir.semaphore, %arg5: !ttir.semaphore, %arg6: !ttir.semaphore) attributes {ttir.thread = #ttir.thread<datamovement>} {
    return
  }

  func.func private @datamovement_kernel1(%arg0: memref<1x3x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!tt.tile<32x32, f32>, #l1_>, %arg3: !ttir.semaphore, %arg4: !ttir.semaphore, %arg5: !ttir.semaphore, %arg6: !ttir.semaphore) attributes {ttir.thread = #ttir.thread<datamovement>} {
    return
  }

  func.func private @compute_kernel2(%arg0: memref<1x3x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!tt.tile<32x32, f32>, #l1_>, %arg3: !ttir.semaphore, %arg4: !ttir.semaphore, %arg5: !ttir.semaphore, %arg6: !ttir.semaphore) attributes {ttir.thread = #ttir.thread<compute>} {
    return
  }
}
