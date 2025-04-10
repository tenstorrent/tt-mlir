// RUN: ttmlir-opt --convert-ttir-to-ttmetal %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #tt.memory_space<l1>
#system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, physical_cores = {worker = [ 18x18,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  1x0,  1x1,  1x2,  1x3,  1x4,  1x5,  1x6,  1x7,  2x0,  2x1,  2x2,  2x3,  2x4,  2x5,  2x6,  2x7,  3x0,  3x1,  3x2,  3x3,  3x4,  3x5,  3x6,  3x7,  4x0,  4x1,  4x2,  4x3,  4x4,  4x5,  4x6,  4x7,  5x0,  5x1,  5x2,  5x3,  5x4,  5x5,  5x6,  5x7,  6x0,  6x1,  6x2,  6x3,  6x4,  6x5,  6x6,  6x7,  7x0,  7x1,  7x2,  7x3,  7x4,  7x5,  7x6,  7x7] dram = [ 8x0,  9x0,  10x0,  8x1,  9x1,  10x1,  8x2,  9x2,  10x2,  8x3,  9x3,  10x3]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>
module attributes {tt.system_desc = #system_desc} {
  tt.device @default_device = <workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @generic0(%arg0: memref<1x1x8x24x!tt.tile<32x32, f32>, #tt.shard<98304x4096>, #l1_>, %arg1: memref<1x1x24x32x!tt.tile<32x32, f32>, #tt.shard<131072x4096>, #l1_>) -> memref<1x1x8x32x!tt.tile<32x32, f32>, #tt.shard<131072x4096>, #l1_> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<8x8x1x4x!tt.tile<32x32, f32>, #l1_>
    %alloc_0 = memref.alloc() {alignment = 64 : i64, address = 0x10000} : memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.shard<12288x4096>, #l1_>
    %stream = "ttir.stream_layout"(%arg0, %alloc_0) : (memref<1x1x8x24x!tt.tile<32x32, f32>, #tt.shard<98304x4096>, #l1_>, memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.shard<12288x4096>, #l1_>) -> memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 12288 + d3 * 4096)>, #l1_>
    %alloc_1 = memref.alloc() {alignment = 64 : i64, address = 0x15000} : memref<8x8x3x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>
    %stream_2 = "ttir.stream_layout"(%arg1, %alloc_1) : (memref<1x1x24x32x!tt.tile<32x32, f32>, #tt.shard<131072x4096>, #l1_>, memref<8x8x3x4x!tt.tile<32x32, f32>, #tt.shard<16384x4096>, #l1_>) -> memref<8x8x3x4x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 16384 + d3 * 4096)>, #l1_>
    // CHECK: "ttmetal.enqueue_program"
    // CHECK-SAME: {{.*}}core_ranges = [#ttmetal.core_range<0x0, 8x8>, #ttmetal.core_range<0x0, 8x8>, #ttmetal.core_range<0x0, 8x8>]
    // CHECK-SAME: {{.*}}threads = [#ttir.thread<datamovement, @datamovement_kernel0>, #ttir.thread<datamovement, @datamovement_kernel1>, #ttir.thread<compute, @compute_kernel2>]
    ttir.generic {grid = #tt.grid<8x8>, indexing_maps = [], iterator_types = [], threads = [#ttir.thread<datamovement, @datamovement_kernel0>, #ttir.thread<datamovement, @datamovement_kernel1>, #ttir.thread<compute, @compute_kernel2>]}
        ins(%stream, %stream_2 : memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 12288 + d3 * 4096)>, #l1_>, memref<8x8x3x4x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 16384 + d3 * 4096)>, #l1_>)
        outs(%alloc : memref<8x8x1x4x!tt.tile<32x32, f32>, #l1_>)
    %view = "ttir.view_layout"(%alloc) : (memref<8x8x1x4x!tt.tile<32x32, f32>, #l1_>) -> memref<1x1x8x32x!tt.tile<32x32, f32>, #tt.shard<131072x4096>, #l1_>
    return %view : memref<1x1x8x32x!tt.tile<32x32, f32>, #tt.shard<131072x4096>, #l1_>
  }

  func.func private @datamovement_kernel0(%arg0: memref<1x3x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!tt.tile<32x32, f32>, #l1_>, %arg3: !ttir.semaphore, %arg4: !ttir.semaphore, %arg5: !ttir.semaphore, %arg6: !ttir.semaphore) attributes {ttir.thread_type = 1 : i32} {
    return
  }

  func.func private @datamovement_kernel1(%arg0: memref<1x3x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!tt.tile<32x32, f32>, #l1_>, %arg3: !ttir.semaphore, %arg4: !ttir.semaphore, %arg5: !ttir.semaphore, %arg6: !ttir.semaphore) attributes {ttir.thread_type = 1 : i32} {
    return
  }

  func.func private @compute_kernel2(%arg0: memref<1x3x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!tt.tile<32x32, f32>, #l1_>, %arg3: !ttir.semaphore, %arg4: !ttir.semaphore, %arg5: !ttir.semaphore, %arg6: !ttir.semaphore) attributes {ttir.thread_type = 1 : i32} {
    return
  }
}
