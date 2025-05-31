// RUN: ttmlir-opt --tt-register-device --ttir-generic-lower-dmas --convert-ttir-to-ttkernel --ttkernel-control-dst-section %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #tt.memory_space<l1>
module {
  func.func private @datamovement_kernel0(%arg0: memref<1x3x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!tt.tile<32x32, f32>, #l1_>, %arg3: !ttir.semaphore, %arg4: !ttir.semaphore) attributes {ttir.thread = #ttir.thread<datamovement>} {
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
        %1 = ttir.get_global_operand(0) : memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.shard<12288x4096>, #l1_>
        // CHECK: %{{[0-9]+}} = "ttkernel.get_noc_addr"
        // CHECK: %{{[0-9]+}} = "ttkernel.get_write_ptr"
        // CHECK: "ttkernel.noc_async_read"
        %tx = ttir.dma %1 [%c0, %arg7, %c0, %c0], %arg0 [%c0, %c0] : (memref<8x8x1x3x!tt.tile<32x32, f32>, #tt.shard<12288x4096>, #l1_>, memref<1x3x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
        // CHECK: "ttkernel.noc_async_read_barrier"
        ttir.dma_wait %tx
        // CHECK: %{{[0-9]+}} = "ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>"
        // CHECK: "ttkernel.noc_semaphore_wait"
        // CHECK: "ttkernel.noc_semaphore_set"
        ttir.semaphore_wait %arg3, %c7 reset %c0
        // CHECK: %{{[0-9]+}} = "ttkernel.get_read_ptr"
        // CHECK: %{{[0-9]+}} = "ttkernel.get_write_ptr"
        // CHECK: %{{[0-9]+}} = "ttkernel.get_noc_multicast_addr"
        // CHECK: "ttkernel.noc_async_write_multicast"
        %tx_0 = ttir.dma %arg0 [%c0, %c0], %arg0 [%c0, %c0] core[%core0, %c0] mcast[%c1, %c8] : (memref<1x3x!tt.tile<32x32, f32>, #l1_>, memref<1x3x!tt.tile<32x32, f32>, #l1_>) -> !ttir.mem_tx
        // CHECK: "ttkernel.noc_async_write_barrier"
        ttir.dma_wait %tx_0
        // CHECK: %{{[0-9]+}} = "ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>"
        // CHECK: "ttkernel.noc_semaphore_set"
        // CHECK: "ttkernel.noc_semaphore_set_multicast"
        ttir.semaphore_set %arg4, %c1, core[%core0, %c0] mcast[%c1, %c8]
      } else {
        // CHECK: %{{[0-9]+}} = "ttkernel.get_noc_addr"
        // CHECK: "ttkernel.noc_semaphore_inc"
        ttir.semaphore_inc %arg3, %c1, core[%core0, %c0]
        // CHECK: %{{[0-9]+}} = "ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>"
        // CHECK: "ttkernel.noc_semaphore_wait"
        // CHECK: "ttkernel.noc_semaphore_set"
        ttir.semaphore_wait %arg4, %c1 reset %c0
      }
      // CHECK: "ttkernel.cb_push_back"
      ttir.yield %arg0 : (memref<1x3x!tt.tile<32x32, f32>, #l1_>)
    }
    return
  }

  func.func private @compute_kernel2(%arg0: memref<1x3x!tt.tile<32x32, f32>, #l1_>, %arg1: memref<3x4x!tt.tile<32x32, f32>, #l1_>, %arg2: memref<1x4x!tt.tile<32x32, f32>, #l1_>, %arg3: !ttir.semaphore, %arg4: !ttir.semaphore) attributes {ttir.thread = #ttir.thread<compute>} {
    %c4 = arith.constant 4 : index
    %c3 = arith.constant 3 : index
    %c1 = arith.constant 1 : index
    %c0 = arith.constant 0 : index
    // CHECK: "ttkernel.cb_reserve_back"
    // CHECK: "ttkernel.cb_reinterpret_shape"
    // CHECK: "ttkernel.cb_reinterpret_shape"
    // CHECK: "ttkernel.cb_reinterpret_shape"
    %collapse_shape = memref.collapse_shape %arg0 [[0, 1]] : memref<1x3x!tt.tile<32x32, f32>, #l1_> into memref<3x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_0 = memref.collapse_shape %arg1 [[0, 1]] : memref<3x4x!tt.tile<32x32, f32>, #l1_> into memref<12x!tt.tile<32x32, f32>, #l1_>
    %collapse_shape_1 = memref.collapse_shape %arg2 [[0, 1]] : memref<1x4x!tt.tile<32x32, f32>, #l1_> into memref<4x!tt.tile<32x32, f32>, #l1_>
    // CHECK: "ttkernel.mm_init"
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
          // CHECK: "ttkernel.mm_init_short"
          // CHECK: "ttkernel.matmul_tiles"
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
