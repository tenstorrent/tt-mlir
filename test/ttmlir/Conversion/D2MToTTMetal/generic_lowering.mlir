// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel --convert-d2m-to-ttmetal -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #ttcore.memory_space<l1>
module {
  func.func @generic0(%arg0: memref<1x1x8x24x!ttcore.tile<32x32, f32>, #ttcore.shard<98304x4096>, #l1_>, %arg1: memref<1x1x24x32x!ttcore.tile<32x32, f32>, #ttcore.shard<131072x4096>, #l1_>) -> memref<8x8x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_> {
    %alloc = memref.alloc() {alignment = 64 : i64, address = 0x1000} : memref<8x8x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
    %alloc_0 = memref.alloc() {alignment = 64 : i64, address = 0x10000} : memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096>, #l1_>
    %stream = "d2m.stream_layout"(%arg0, %alloc_0) : (memref<1x1x8x24x!ttcore.tile<32x32, f32>, #ttcore.shard<98304x4096>, #l1_>, memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096>, #l1_>) -> memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
    %alloc_1 = memref.alloc() {alignment = 64 : i64, address = 0x15000} : memref<8x8x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
    %stream_2 = "d2m.stream_layout"(%arg1, %alloc_1) : (memref<1x1x24x32x!ttcore.tile<32x32, f32>, #ttcore.shard<131072x4096>, #l1_>, memref<8x8x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> memref<8x8x3x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
    // CHECK: "ttmetal.enqueue_program"
    // CHECK-SAME: {{.*}}cb_ports = array<i64: 0, 1, 2>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel0, #ttmetal.core_range<0x0, 8x8>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>, <semaphore[0]>, <semaphore[1]>, <semaphore[2]>, <semaphore[3]>]>, noc0>, #ttmetal.noc_config<@datamovement_kernel1, #ttmetal.core_range<0x0, 8x8>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>, <semaphore[0]>, <semaphore[1]>, <semaphore[2]>, <semaphore[3]>]>, noc1>, #ttmetal.compute_config<@compute_kernel2, #ttmetal.core_range<0x0, 8x8>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, hifi4, true, false, false, [default]>]
    d2m.generic {block_factors = [], grid = #ttcore.grid<8x8>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @datamovement_kernel0>, #d2m.thread<datamovement, @datamovement_kernel1>, #d2m.thread<compute, @compute_kernel2>]}
        ins(%stream, %stream_2 : memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<8x8x3x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>)
        outs(%alloc : memref<8x8x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>)
    return %alloc  : memref<8x8x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  }

  func.func private @datamovement_kernel0(%arg0: memref<1x3x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: !d2m.semaphore, %arg4: !d2m.semaphore, %arg5: !d2m.semaphore, %arg6: !d2m.semaphore) attributes {d2m.thread = #d2m.thread<datamovement>} {
    return
  }

  func.func private @datamovement_kernel1(%arg0: memref<1x3x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: !d2m.semaphore, %arg4: !d2m.semaphore, %arg5: !d2m.semaphore, %arg6: !d2m.semaphore) attributes {d2m.thread = #d2m.thread<datamovement>} {
    return
  }

  func.func private @compute_kernel2(%arg0: memref<1x3x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: !d2m.semaphore, %arg4: !d2m.semaphore, %arg5: !d2m.semaphore, %arg6: !d2m.semaphore) attributes {d2m.thread = #d2m.thread<compute>} {
    return
  }
}
