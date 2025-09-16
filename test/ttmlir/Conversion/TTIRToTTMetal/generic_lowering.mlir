// RUN: ttmlir-opt --ttcore-register-device --convert-ttir-to-ttkernel --convert-ttir-to-ttmetal -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #ttcore.memory_space<l1>
module {
  func.func @generic0(%arg0: memref<1x1x8x24x!ttcore.tile<32x32, f32>, #ttcore.shard<98304x4096>, #l1_>, %arg1: memref<1x1x24x32x!ttcore.tile<32x32, f32>, #ttcore.shard<131072x4096>, #l1_>) -> memref<8x8x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_> {
    %alloc = memref.alloc() {alignment = 64 : i64, address = 0x1000} : memref<8x8x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
    %alloc_0 = memref.alloc() {alignment = 64 : i64, address = 0x10000} : memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096>, #l1_>
    %stream = "ttir.stream_layout"(%arg0, %alloc_0) : (memref<1x1x8x24x!ttcore.tile<32x32, f32>, #ttcore.shard<98304x4096>, #l1_>, memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096>, #l1_>) -> memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
    %alloc_1 = memref.alloc() {alignment = 64 : i64, address = 0x15000} : memref<8x8x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
    %stream_2 = "ttir.stream_layout"(%arg1, %alloc_1) : (memref<1x1x24x32x!ttcore.tile<32x32, f32>, #ttcore.shard<131072x4096>, #l1_>, memref<8x8x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>) -> memref<8x8x3x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>
    // CHECK: "ttmetal.enqueue_program"
    // CHECK-SAME: {{.*}}cb_ports = array<i64: 0, 1, 2>, kernelConfigs = [#ttmetal.noc_config<@datamovement_kernel0, #ttmetal.core_range<0x0, 8x8>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>, <semaphore[0]>, <semaphore[1]>, <semaphore[2]>, <semaphore[3]>]>, noc0>, #ttmetal.noc_config<@datamovement_kernel1, #ttmetal.core_range<0x0, 8x8>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>, <semaphore[0]>, <semaphore[1]>, <semaphore[2]>, <semaphore[3]>]>, noc1>, #ttmetal.compute_config<@compute_kernel2, #ttmetal.core_range<0x0, 8x8>, #ttmetal.kernel_args< ct_args = [<cb_port[0]>, <cb_port[1]>, <cb_port[2]>]>, hifi4, false, true, false, [default]>]
    ttir.generic {block_factors = [], grid = #ttcore.grid<8x8>, indexing_maps = [], iterator_types = [], threads = [#ttir.thread<datamovement, @datamovement_kernel0>, #ttir.thread<datamovement, @datamovement_kernel1>, #ttir.thread<compute, @compute_kernel2>]}
        ins(%stream, %stream_2 : memref<8x8x1x3x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>, memref<8x8x3x4x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1_>)
        outs(%alloc : memref<8x8x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>)
    return %alloc  : memref<8x8x1x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096>, #l1_>
  }

  func.func private @datamovement_kernel0(%arg0: memref<1x3x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: !ttir.semaphore, %arg4: !ttir.semaphore, %arg5: !ttir.semaphore, %arg6: !ttir.semaphore) attributes {ttir.thread = #ttir.thread<datamovement>} {
    return
  }

  func.func private @datamovement_kernel1(%arg0: memref<1x3x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: !ttir.semaphore, %arg4: !ttir.semaphore, %arg5: !ttir.semaphore, %arg6: !ttir.semaphore) attributes {ttir.thread = #ttir.thread<datamovement>} {
    return
  }

  func.func private @compute_kernel2(%arg0: memref<1x3x!ttcore.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!ttcore.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!ttcore.tile<32x32, f32>, #l1_>, %arg3: !ttir.semaphore, %arg4: !ttir.semaphore, %arg5: !ttir.semaphore, %arg6: !ttir.semaphore) attributes {ttir.thread = #ttir.thread<compute>} {
    return
  }
}
