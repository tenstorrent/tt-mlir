// RUN: ttmlir-opt --split-input-file --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s
// Test for lowering kernel arguments from D2M dialect to TTKernel dialect

// Test for lowering cb types.
#l1 = #ttcore.memory_space<l1>
// CHECK: ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>]>
func.func private @cb_lowering(%arg0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %arg1: !d2m.cb<memref<32x32xf32, #l1>>) attributes {d2m.thread = #d2m.thread<compute>, tt.function_type = "kernel"} {
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

// Test for lowering scalar types.
#l1 = #ttcore.memory_space<l1>
// CHECK: ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = scalar, operand_index = 2>]>
func.func private @scalar_lowering(%cb0: !d2m.cb<memref<32x32xf32, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %scalar: !d2m.scalar<ui32>) attributes {d2m.thread = #d2m.thread<datamovement>, tt.function_type = "kernel"} {
  d2m.print_arg %scalar : !d2m.scalar<ui32>
  %0 = d2m.reserve %cb0 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
  d2m.push %cb0 : <memref<32x32xf32, #l1>>
  %1 = d2m.wait %cb0 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
  %2 = d2m.reserve %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  %3 = "d2m.tile_tilize_block"(%1, %2) : (memref<32x32xf32, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  d2m.pop %cb0 : <memref<32x32xf32, #l1>>
  d2m.push %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
  %5 = d2m.wait %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  d2m.pop %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
  return
}

// -----

// Test for lowering global semaphore types.
#l1 = #ttcore.memory_space<l1>
// CHECK: ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = global_semaphore, operand_index = 2>, <arg_type = global_semaphore, operand_index = 3>, <arg_type = buffer_address, operand_index = 1>]>
func.func private @datamovement_kernel1(%arg0: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>, %arg1: !d2m.cb<memref<2x2x!ttcore.tile<32x32, f32>, #l1>>, %arg2: !d2m.global_semaphore, %arg3: !d2m.global_semaphore) attributes {d2m.thread = #d2m.thread<datamovement>, tt.function_type = "kernel"} {
  // CHECK: %2 = ttkernel.get_compile_time_arg_val(2) : () -> !ttkernel.global_semaphore
  // CHECK: %3 = ttkernel.get_compile_time_arg_val(3) : () -> !ttkernel.global_semaphore
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
  // CHECK: %4 = ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>(%2) : (!ttkernel.global_semaphore) -> !ttkernel.l1_addr_ptr
  d2m.semaphore_wait %arg2, %c1 : !d2m.global_semaphore
  return
}

// -----

// Test for lowering local semaphore types.
#l1 = #ttcore.memory_space<l1>
#map1 = affine_map<() -> ()>

// CHECK: ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>, <arg_type = local_semaphore, operand_index = 3>, <arg_type = local_semaphore, operand_index = 4>, <arg_type = local_semaphore, operand_index = 5>, <arg_type = local_semaphore, operand_index = 6>, <arg_type = buffer_address, operand_index = 0>]>
func.func private @datamovement_kernel4(%arg0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %arg1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %arg2: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %arg3: !d2m.local_semaphore, %arg4: !d2m.local_semaphore, %arg5: !d2m.local_semaphore, %arg6: !d2m.local_semaphore) attributes {d2m.thread = #d2m.thread<datamovement>, tt.function_type = "kernel"} {
  // CHECK: %4 = ttkernel.get_semaphore(%3) : (i32) -> !ttkernel.local_semaphore
  // CHECK: %6 = ttkernel.get_semaphore(%5) : (i32) -> !ttkernel.local_semaphore
  // CHECK: %8 = ttkernel.get_semaphore(%7) : (i32) -> !ttkernel.local_semaphore
  // CHECK: %10 = ttkernel.get_semaphore(%9) : (i32) -> !ttkernel.local_semaphore
  
  %c4 = arith.constant 4 : index
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  %c7 = arith.constant 7 : index
  %core0 = d2m.core_index(0) {phys_to_virt_map = #map1} : index
  %core1 = d2m.core_index(1) {phys_to_virt_map = #map1} : index
  %0 = arith.cmpi eq, %core1, %c0 : index
  // CHECK: %26 = ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>(%4) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
  // CHECK: %53 = ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>(%6) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
  // CHECK: %19 = ttkernel.reinterpret_cast<volatile tt_l1_ptr uint32_t*>(%6) : (!ttkernel.local_semaphore) -> !ttkernel.l1_addr_ptr
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


// -----

// Test for lowering d2m.reinterpret_cast of scalar types used as kernel arguments.
#l1 = #ttcore.memory_space<l1>
module {
  func.func @test(%arg0: i32, %arg1: si32, %arg2: ui16, %arg3: si16, %arg4: ui8, %arg5: si8, %arg6: i1, %arg7: f32, %arg8: bf16, %arg9: f16) {
    %alloc = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @datamovement0>]}
        ins(%alloc_0 : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>)
        outs(%alloc : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        additionalArgs(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : i32, si32, ui16, si16, ui8, si8, i1, f32, bf16, f16)
     
    return
  }
  func.func private @datamovement0(%arg0: !d2m.cb<memref<32x32xf32, #l1>>, %arg1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %arg2: !d2m.scalar<ui32>, %arg3: !d2m.scalar<ui32>, %arg4: !d2m.scalar<ui32>, %arg5: !d2m.scalar<ui32>, %arg6: !d2m.scalar<ui32>, %arg7: !d2m.scalar<ui32>, %arg8: !d2m.scalar<ui32>, %arg9: !d2m.scalar<ui32>, %arg10: !d2m.scalar<ui32>, %arg11: !d2m.scalar<ui32>) attributes {d2m.thread = #d2m.thread<datamovement>, tt.function_type = "kernel"} {
    // CHECK: %12 = ttkernel.reinterpret_cast %11 : !ttkernel.scalar<ui32> to !ttkernel.scalar<f16>
    // CHECK: %13 = ttkernel.reinterpret_cast %10 : !ttkernel.scalar<ui32> to !ttkernel.scalar<bf16>
    // CHECK: %14 = ttkernel.reinterpret_cast %9 : !ttkernel.scalar<ui32> to !ttkernel.scalar<f32>
    // CHECK: %15 = ttkernel.reinterpret_cast %8 : !ttkernel.scalar<ui32> to !ttkernel.scalar<i1>
    // CHECK: %16 = ttkernel.reinterpret_cast %7 : !ttkernel.scalar<ui32> to !ttkernel.scalar<si8>
    // CHECK: %17 = ttkernel.reinterpret_cast %6 : !ttkernel.scalar<ui32> to !ttkernel.scalar<ui8>
    // CHECK: %18 = ttkernel.reinterpret_cast %5 : !ttkernel.scalar<ui32> to !ttkernel.scalar<si16>
    // CHECK: %19 = ttkernel.reinterpret_cast %4 : !ttkernel.scalar<ui32> to !ttkernel.scalar<ui16>
    // CHECK: %20 = ttkernel.reinterpret_cast %3 : !ttkernel.scalar<ui32> to !ttkernel.scalar<si32>
    // CHECK: %21 = ttkernel.reinterpret_cast %2 : !ttkernel.scalar<ui32> to !ttkernel.scalar<i32>
    %0 = d2m.reinterpret_cast %arg11 : !d2m.scalar<ui32> to !d2m.scalar<f16>
    %1 = d2m.reinterpret_cast %arg10 : !d2m.scalar<ui32> to !d2m.scalar<bf16>
    %2 = d2m.reinterpret_cast %arg9 : !d2m.scalar<ui32> to !d2m.scalar<f32>
    %3 = d2m.reinterpret_cast %arg8 : !d2m.scalar<ui32> to !d2m.scalar<i1>
    %4 = d2m.reinterpret_cast %arg7 : !d2m.scalar<ui32> to !d2m.scalar<si8>
    %5 = d2m.reinterpret_cast %arg6 : !d2m.scalar<ui32> to !d2m.scalar<ui8>
    %6 = d2m.reinterpret_cast %arg5 : !d2m.scalar<ui32> to !d2m.scalar<si16>
    %7 = d2m.reinterpret_cast %arg4 : !d2m.scalar<ui32> to !d2m.scalar<ui16>
    %8 = d2m.reinterpret_cast %arg3 : !d2m.scalar<ui32> to !d2m.scalar<si32>
    %9 = d2m.reinterpret_cast %arg2 : !d2m.scalar<ui32> to !d2m.scalar<i32>
    %10 = d2m.reserve %arg0 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
    d2m.push %arg0 : <memref<32x32xf32, #l1>>
    %11 = d2m.wait %arg0 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
    %12 = d2m.reserve %arg1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    %13 = "d2m.tile_tilize_block"(%11, %12) : (memref<32x32xf32, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    d2m.pop %arg0 : <memref<32x32xf32, #l1>>
    d2m.push %arg1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
    %14 = d2m.wait %arg1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    d2m.pop %arg1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
    return
  }
}