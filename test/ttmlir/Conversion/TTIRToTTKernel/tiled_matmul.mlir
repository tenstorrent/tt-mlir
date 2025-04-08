// RUN: ttmlir-opt --convert-ttir-to-ttkernel --ttmetal-control-dst-section %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #tt.memory_space<l1>
#system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, physical_cores = {worker = [ 18x18,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  1x0,  1x1,  1x2,  1x3,  1x4,  1x5,  1x6,  1x7,  2x0,  2x1,  2x2,  2x3,  2x4,  2x5,  2x6,  2x7,  3x0,  3x1,  3x2,  3x3,  3x4,  3x5,  3x6,  3x7,  4x0,  4x1,  4x2,  4x3,  4x4,  4x5,  4x6,  4x7,  5x0,  5x1,  5x2,  5x3,  5x4,  5x5,  5x6,  5x7,  6x0,  6x1,  6x2,  6x3,  6x4,  6x5,  6x6,  6x7,  7x0,  7x1,  7x2,  7x3,  7x4,  7x5,  7x6,  7x7] dram = [ 8x0,  9x0,  10x0,  8x1,  9x1,  10x1,  8x2,  9x2,  10x2,  8x3,  9x3,  10x3]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [3 : i32], [ 0x0x0x0]>
module attributes {tt.system_desc = #system_desc} {
    tt.device @default_device = <workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5] -> (0, 0, (((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv s4) mod 12, ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) floordiv (s4 * 12) + ((d0 * s1) * (s2 * s3) + d1 * (s2 * s3) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
    func.func private @datamovement_kernel0(%arg0: memref<1x3x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!tt.tile<32x32, f32>, #l1_>, %arg3: !ttir.semaphore, %arg4: !ttir.semaphore, %arg5: !ttir.semaphore, %arg6: !ttir.semaphore) attributes {ttir.thread_type = 1 : i32} {
    %c8 = arith.constant 8 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %c7 = arith.constant 7 : index
    // CHECK: %{{[0-9]+}} = "ttkernel.my_x"
    %core1 = ttir.core_index(1) : index
    // CHECK: %{{[0-9]+}} = "ttkernel.my_y"
    %core0 = ttir.core_index(0) : index
    %0 = arith.cmpi eq, %core1, %c0 : index
    scf.for %arg7 = %c0 to %c8 step %c1 {
      // CHECK: "ttkernel.cb_reserve_back"
      scf.if %0 {
        // CHECK: %{{[0-9]+}} = "ttkernel.get_compile_time_arg_val"
        %1 = ttir.get_global_operand(0) : memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 12288 + d3 * 4096)>, #l1_>
        // CHECK: %{{[0-9]+}} = "ttkernel.get_noc_addr_xy"
        // CHECK: "ttkernel.noc_async_read"
        %tx = ttir.dma %1 [%c0, %arg7], %arg0 : (memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.stream<(d0, d1, d2, d3) -> (d0, d1, d2 * 12288 + d3 * 4096)>, #l1_>, memref<1x3x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
        // CHECK: "ttkernel.noc_async_read_barrier"
        ttir.dma_wait %tx
        // ttir.semaphore_wait %arg3, %c7 reset %c0
        // CHECK: %{{[0-9]+}} = "ttkernel.get_write_ptr"
        // CHECK: %{{[0-9]+}} = "ttkernel.get_noc_multicast_addr"
        // CHECK: "ttkernel.noc_async_write_multicast"
        %tx_0 = ttir.dma %arg0, %arg0 core[%core0, %c0] mcast[%c1, %c8] : (memref<1x3x!tt.tile<32x32, f32>, #l1_>, memref<1x3x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
        // CHECK: "ttkernel.noc_async_write_barrier"
        ttir.dma_wait %tx_0
        // ttir.semaphore_set %arg4, %c1, core[%core0, %c0] mcast[%c1, %c8]
      } else {
        // ttir.semaphore_inc %arg3, %c1, core[%core0, %c0]
        // ttir.semaphore_wait %arg4, %c1 reset %c0
      }
      // CHECK: "ttkernel.cb_push_back"
      ttir.yield %arg0 : (memref<1x3x!tt.tile<32x32, f32>, #l1_>)
    }
    return
  }

  func.func private @compute_kernel2(%arg0: memref<1x3x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!tt.tile<32x32, f32>, #l1_>, %arg3: !ttir.semaphore, %arg4: !ttir.semaphore, %arg5: !ttir.semaphore, %arg6: !ttir.semaphore) attributes {ttir.thread_type = 0 : i32} {
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x3x!tt.tile<32x32, f32>, #l1_> into memref<3x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<3x4x!tt.tile<32x32, f32>, #l1_> into memref<12x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<1x4x!tt.tile<32x32, f32>, #l1_> into memref<4x!tt.tile<32x32, f32>, #l1_>
    // CHECK: "ttkernel.cb_reserve_back"
    // CHECK: "ttkernel.cb_wait_front"
    // CHECK: "ttkernel.cb_wait_front"
    ttir.await %arg0, %arg1 : (memref<1x3x!tt.tile<32x32, f32>, #l1_>, memref<3x4x!tt.tile<32x32, f32>, #l1_>)
    scf.for %arg7 = %c0 to %c1 step %c1 {
      scf.for %arg8 = %c0 to %c4 step %c1 {
        scf.for %arg9 = %c0 to %c3 step %c1 {
          // CHECK: "ttkernel.tile_regs_acquire"
          %0 = arith.muli %arg7, %c1 overflow<nsw> : index
          %1 = arith.addi %0, %arg9 : index
          %2 = memref.load %collapse_shape[%1] : memref<3x!tt.tile<32x32, f32>, #l1_>
          %3 = arith.muli %arg9, %c4 overflow<nsw> : index
          %4 = arith.addi %3, %arg8 : index
          %5 = memref.load %collapse_shape_0[%4] : memref<12x!tt.tile<32x32, f32>, #l1_>
          %6 = arith.muli %arg7, %c4 overflow<nsw> : index
          %7 = arith.addi %6, %arg8 : index
          // CHECK: "ttkernel.copy_tile_init"
          // CHECK: "ttkernel.copy_tile"
          %8 = memref.load %collapse_shape_1[%7] : memref<4x!tt.tile<32x32, f32>, #l1_>
          // CHECK: "ttkernel.matmul_tiles"
          // CHECK-SAME: {{.*}}%c0{{.*}}%c1
          // CHECK: "ttkernel.tile_regs_commit"
          %9 = "ttir.tile_matmul"(%2, %5, %8) : (!tt.tile<32x32, f32>, !tt.tile<32x32, f32>, !tt.tile<32x32, f32>) -> !tt.tile<32x32, f32>
          %10 = arith.muli %arg7, %c4 overflow<nsw> : index
          %11 = arith.addi %10, %arg8 : index
          // CHECK: "ttkernel.tile_regs_wait"
          // CHECK: "ttkernel.pack_tile"
          // CHECK: "ttkernel.tile_regs_release"
          memref.store %9, %collapse_shape_1[%11] : memref<4x!tt.tile<32x32, f32>, #l1_>
        }
      }
    }
    // CHECK: "ttkernel.cb_push_back"
    // CHECK: "ttkernel.cb_wait_front"
    // CHECK: "ttkernel.cb_pop_front"
    // CHECK: "ttkernel.cb_pop_front"
    // CHECK: "ttkernel.cb_pop_front"
    ttir.yield %arg2 : (memref<1x4x!tt.tile<32x32, f32>, #l1_>)
    ttir.await %arg2 : (memref<1x4x!tt.tile<32x32, f32>, #l1_>)
    return
  }
}
