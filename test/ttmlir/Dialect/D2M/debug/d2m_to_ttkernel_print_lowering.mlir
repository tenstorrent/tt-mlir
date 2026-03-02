// RUN: ttmlir-opt --split-input-file --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s
// Test for lowering debug operations in D2M dialect to TTKernel dialect

// Test for lowering scalar print operations.
#l1 = #ttcore.memory_space<l1>
func.func private @datamovement0(%arg0: !d2m.cb<memref<32x32xf32, #l1>>, %arg1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %arg2: !d2m.scalar<ui32>) attributes {d2m.thread = #d2m.thread<datamovement>, tt.function_type = "kernel"} {
  // CHECK: %2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.scalar<ui32>
  // CHECK: ttkernel.dprint("Printing arg: {}", %2) : (!ttkernel.scalar<ui32>) -> ()
  d2m.print_arg %arg2 : !d2m.scalar<ui32>
  %2 = d2m.reserve %arg0 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
  d2m.push %arg0 : <memref<32x32xf32, #l1>>
  %3 = d2m.wait %arg0 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
  %4 = d2m.reserve %arg1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  %5 = "d2m.tile_tilize_block"(%3, %4) : (memref<32x32xf32, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  d2m.pop %arg0 : <memref<32x32xf32, #l1>>
  d2m.push %arg1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
  %6 = d2m.wait %arg1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  d2m.pop %arg1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
  return
}

// -----

// Test for lowering cb print operations.
#l1 = #ttcore.memory_space<l1>
func.func private @cb_lowering(%arg0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %arg1: !d2m.cb<memref<32x32xf32, #l1>>) attributes {d2m.thread = #d2m.thread<compute>, tt.function_type = "kernel"} {
  // CHECK: %0 = ttkernel.get_compile_time_arg_val(0) : () -> !ttkernel.cb<1, !ttcore.tile<32x32, f32>>
  // CHECK: ttkernel.dprint("Printing arg: {}", %0) : (!ttkernel.cb<1, !ttcore.tile<32x32, f32>>) -> ()
  d2m.print_arg %arg0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
  %0 = d2m.reserve %arg0 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  d2m.push %arg0 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
  %1 = d2m.wait %arg0 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  %2 = d2m.reserve %arg1 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
  %3 = "d2m.tile_untilize_block"(%1, %2) : (memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<32x32xf32, #l1>) -> memref<32x32xf32, #l1>
  d2m.pop %arg0 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
  d2m.push %arg1 : <memref<32x32xf32, #l1>>
  %4 = d2m.wait %arg1 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
  d2m.pop %arg1 : <memref<32x32xf32, #l1>>
  return
}

// -----

// Test for lowering global semaphore print operations.
#l1 = #ttcore.memory_space<l1>
func.func private @datamovement_kernel1(%arg0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>, %arg1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>, %arg2: !d2m.global_semaphore, %arg3: !d2m.global_semaphore) attributes {d2m.thread = #d2m.thread<datamovement>, tt.function_type = "kernel"} {
  // CHECK: %2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.global_semaphore
  // CHECK: ttkernel.dprint("Printing arg: {}", %2) : (!ttkernel.global_semaphore) -> ()
  d2m.print_arg %arg2 : !d2m.global_semaphore
  %c2 = arith.constant 2 : index
  %c1 = arith.constant 1 : index
  %c0 = arith.constant 0 : index
  scf.for %arg4 = %c0 to %c2 step %c1 {
    scf.for %arg5 = %c0 to %c2 step %c1 {
      %0 = d2m.wait %arg1 : <memref<2x2x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x2x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.get_global_operand(1) : memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>
      %tx = d2m.dma_write %0[%c0], %1[%arg4, %arg5, %c0], <4> : (memref<2x2x!ttcore.tile<32x32, f32>, #l1>, memref<2x2x2x2x!ttcore.tile<32x32, f32>, #ttcore.shard<8192x4096, 1>, #l1>) -> !d2m.mem_tx
      d2m.dma_wait %tx
      d2m.pop %arg1 : <memref<2x2x!ttcore.tile<32x32, f32>, #l1>>
    }
  }
  d2m.semaphore_wait %arg2, %c1 : !d2m.global_semaphore
  return
}

// -----

// Test for lowering local semaphore print operations.
#l1 = #ttcore.memory_space<l1>
#map = affine_map<() -> ()>
module {
  func.func private @datamovement_kernel4(%arg0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %arg1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %arg2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %arg3: !d2m.local_semaphore, %arg4: !d2m.local_semaphore, %arg5: !d2m.local_semaphore, %arg6: !d2m.local_semaphore) attributes {d2m.thread = #d2m.thread<datamovement>, tt.function_type = "kernel"} {
    // CHECK: %4 = ttkernel.get_semaphore(%3) : (i32) -> !ttkernel.local_semaphore
    // CHECK: ttkernel.dprint("Printing arg: {}", %4) : (!ttkernel.local_semaphore) -> ()
    d2m.print_arg %arg3 : !d2m.local_semaphore
    %c4 = arith.constant 4 : index
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c7 = arith.constant 7 : index
    %core0 = d2m.core_index(0) {phys_to_virt_map = #map} : index
    %core1 = d2m.core_index(1) {phys_to_virt_map = #map} : index
    %0 = arith.cmpi eq, %core1, %c0 : index
    scf.for %arg7 = %c0 to %c4 step %c1 {
      %1 = d2m.reserve %arg0 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      scf.if %0 {
        %2 = d2m.get_global_operand(0) : memref<8x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
        %tx = d2m.dma_read %2[%core0, %arg7, %c0], %1[%c0], <1> : (memref<8x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
        d2m.dma_wait %tx
        d2m.semaphore_wait %arg3, %c7 reset %c0 : !d2m.local_semaphore
        %tx_0 = d2m.dma_write %1[%c0], %1[%c0] core[%core0, %c0] mcast[%c1, %c8], <1> : (memref<1x1x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
        d2m.dma_wait %tx_0
        d2m.semaphore_set %arg4, %c1, core[%core0, %c0] mcast[%c1, %c8] : !d2m.local_semaphore
      } else {
        d2m.semaphore_inc %arg3, %c1, core[%core0, %c0] : !d2m.local_semaphore
        d2m.semaphore_wait %arg4, %c1 reset %c0 : !d2m.local_semaphore
      }
      d2m.push %arg0 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
    } {d2m.blocking_loop = 0 : i64}
    return
  }
}
