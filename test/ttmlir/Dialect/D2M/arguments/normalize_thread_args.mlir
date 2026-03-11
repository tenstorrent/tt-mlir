// RUN: ttmlir-opt --split-input-file --normalize-thread-args %s | FileCheck %s
// Test for capturing additional arguments into threads

// Test for capturing aliased cb types.
#l1 = #ttcore.memory_space<l1>
module {
  func.func @test(%arg0: index) {
    %alloc = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_0 : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>)
        outs(%alloc : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
     {
    ^unified0():
      // CHECK: %0 = d2m.get_cb(0) operand_index = 0 resolution_stage =  compile : <memref<32x32xf32, #l1>>
      // CHECK: %1 = d2m.get_cb(1) operand_index = 1 resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb0 = d2m.get_cb(0) operand_index = 0 : <memref<32x32xf32, #l1>>
      %cb1 = d2m.get_cb(1) operand_index = 1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %arg0 {

      }
    }
    return
  }
}

// -----

// Test for capturing scratchpad cb types.
#l1 = #ttcore.memory_space<l1>
module {
  func.func @test(%arg0: index) {
    %alloc = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_0 : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>)
        outs(%alloc : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
     {
    ^unified0():
      // CHECK: %0 = d2m.get_cb(0) operand_index = 0 resolution_stage =  compile : <memref<32x32xf32, #l1>>
      // CHECK: %1 = d2m.get_cb(1) resolution_stage =  compile : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %cb0 = d2m.get_cb(0) operand_index = 0 : <memref<32x32xf32, #l1>>
      %cb1 = d2m.get_cb(1) : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %arg0 {

      }
    }
    return
  }
}

// -----

// Test for capturing global semaphore types.
#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#layout1 = #ttcore.metal_layout<logical_shape = 8x8, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @generic_with_global_semaphore(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> {
    %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %1 = d2m.empty() : tensor<8x8x1x1xui32, #layout1>
    %2 = d2m.create_global_semaphore(%1) {value = 0 : ui32} : tensor<8x8x1x1xui32, #layout1> -> !d2m.global_semaphore
    %3 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %4 = d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%arg0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
        outs(%3 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
        additionalArgs(%2 : !d2m.global_semaphore)
     {
    // CHECK: %5 = d2m.get_arg operand_index = 2 resolution_stage =  compile : !d2m.global_semaphore
    ^unified0():
      %cb0 = d2m.get_cb(0) operand_index = 0 : <tensor<1x1x!ttcore.tile<32x32, f32>>>
      %cb1 = d2m.get_cb(1) operand_index = 1 : <tensor<1x1x!ttcore.tile<32x32, f32>>>
      %c1 = arith.constant 1 : index
      d2m.semaphore_wait %2, %c1 : !d2m.global_semaphore
      d2m.yield %3 : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    d2m.reset_global_semaphore(%2) {value = 0 : ui32} : !d2m.global_semaphore
    return %4 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
  }
}

// -----

// Test for capturing local semaphore types.
#l1 = #ttcore.memory_space<l1>
#map = affine_map<() -> ()>

module {
  func.func @test() {
    %alloc_in = memref.alloc() {address = 1024 : i64, alignment = 16 : i64} : memref<8x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_out = memref.alloc() {address = 5120 : i64, alignment = 16 : i64} : memref<8x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %0 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %1 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %2 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %3 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    d2m.generic {block_factors = [], grid = #ttcore.grid<8x8>, indexing_maps = [], iterator_types = [],
                 threads = [#d2m.thread<datamovement>, #d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%alloc_in : memref<8x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        outs(%alloc_out : memref<8x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        additionalArgs(%0, %1, %2, %3 : !d2m.local_semaphore, !d2m.local_semaphore, !d2m.local_semaphore, !d2m.local_semaphore)
    {
    // CHECK: %4 = d2m.get_arg operand_index = 3 resolution_stage = compile : !d2m.local_semaphore
    // CHECK: %5 = d2m.get_arg operand_index = 2 resolution_stage = compile : !d2m.local_semaphore
    ^datamovement0():
      %cb0 = d2m.get_cb(0) operand_index = 0 : <memref<8x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>>
      %cb1 = d2m.get_cb(1) operand_index = 1 : <memref<8x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>>
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %core0 = d2m.core_index(0) {phys_to_virt_map = #map} : index
      %isSender = arith.cmpi eq, %core0, %c0 : index
      scf.if %isSender {
        d2m.semaphore_wait %0, %c7 reset %c0 : !d2m.local_semaphore
        d2m.semaphore_set %1, %c1, core[%core0, %c0] mcast[%c1, %c8] : !d2m.local_semaphore
      } else {
        %core0_r = d2m.core_index(0) {phys_to_virt_map = #map} : index
        d2m.semaphore_inc %0, %c1, core[%core0_r, %c0] : !d2m.local_semaphore
        d2m.semaphore_wait %1, %c1 reset %c0 : !d2m.local_semaphore
      }
    }, {
    // CHECK: %4 = d2m.get_arg operand_index = 5 resolution_stage = compile : !d2m.local_semaphore
    // CHECK: %5 = d2m.get_arg operand_index = 4 resolution_stage = compile : !d2m.local_semaphore
    ^datamovement1():
      %cb0 = d2m.get_cb(0) operand_index = 0 : <memref<8x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>>
      %cb1 = d2m.get_cb(1) operand_index = 1 : <memref<8x8x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>>
      %c7 = arith.constant 7 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %core1 = d2m.core_index(1) {phys_to_virt_map = #map} : index
      %isSender = arith.cmpi eq, %core1, %c0 : index
      scf.if %isSender {
        d2m.semaphore_wait %2, %c7 reset %c0 : !d2m.local_semaphore
        d2m.semaphore_set %3, %c1, core[%c0, %core1] mcast[%c8, %c1] : !d2m.local_semaphore
      } else {
        %core1_r = d2m.core_index(1) {phys_to_virt_map = #map} : index
        d2m.semaphore_inc %2, %c1, core[%c0, %core1_r] : !d2m.local_semaphore
        d2m.semaphore_wait %3, %c1 reset %c0 : !d2m.local_semaphore
      }
    }, {
    ^compute0():
    }
    return
  }
}

// -----

// Test for capturing scalar types.
#l1 = #ttcore.memory_space<l1>
module {
  func.func @test(%arg0: index) {
    %alloc = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_0 : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>)
        outs(%alloc : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        additionalArgs(%arg0 : index)
     {
    // CHECK: %0 = d2m.get_arg operand_index = 2 resolution_stage = compile : index
    ^unified0():
      %cb0 = d2m.get_cb(0) operand_index = 0 : <memref<32x32xf32, #l1>>
      %cb1 = d2m.get_cb(1) operand_index = 1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      scf.for %arg2 = %c0 to %c1 step %arg0 {

      }
    }
    return
  }
}

// -----

// Test for inserting multiple dtype scalar values.
#l1 = #ttcore.memory_space<l1>
module {
  func.func @test(%arg0: i32, %arg1: si32, %arg2: ui16, %arg3: si16, %arg4: ui8, %arg5: si8, %arg6: i1, %arg7: f32, %arg8: bf16, %arg9: f16) {
    %alloc = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_0 : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>)
        outs(%alloc : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        additionalArgs(%arg0, %arg1, %arg2, %arg3, %arg4, %arg5, %arg6, %arg7, %arg8, %arg9 : i32, si32, ui16, si16, ui8, si8, i1, f32, bf16, f16)
     {
    ^unified0():
      // CHECK: %0 = d2m.get_arg operand_index = 11 resolution_stage =  compile : f16
      // CHECK: %1 = d2m.get_arg operand_index = 10 resolution_stage =  compile : bf16
      // CHECK: %2 = d2m.get_arg operand_index = 9 resolution_stage =  compile : f32
      // CHECK: %3 = d2m.get_arg operand_index = 8 resolution_stage =  compile : i1
      // CHECK: %4 = d2m.get_arg operand_index = 7 resolution_stage =  compile : si8
      // CHECK: %5 = d2m.get_arg operand_index = 6 resolution_stage =  compile : ui8
      // CHECK: %6 = d2m.get_arg operand_index = 5 resolution_stage =  compile : si16
      // CHECK: %7 = d2m.get_arg operand_index = 4 resolution_stage =  compile : ui16
      // CHECK: %8 = d2m.get_arg operand_index = 3 resolution_stage =  compile : si32
      // CHECK: %9 = d2m.get_arg operand_index = 2 resolution_stage =  compile : i32
      // unrealized_conversion_cast are inserted to force the argument to not be canonicalized away
      %temp0 = builtin.unrealized_conversion_cast %arg0 : i32 to i32
      %temp1 = builtin.unrealized_conversion_cast %arg1 : si32 to si32
      %temp2 = builtin.unrealized_conversion_cast %arg2 : ui16 to ui16
      %temp3 = builtin.unrealized_conversion_cast %arg3 : si16 to si16
      %temp4 = builtin.unrealized_conversion_cast %arg4 : ui8 to ui8
      %temp5 = builtin.unrealized_conversion_cast %arg5 : si8 to si8
      %temp6 = builtin.unrealized_conversion_cast %arg6 : i1 to i1
      %temp7 = builtin.unrealized_conversion_cast %arg7 : f32 to f32
      %temp8 = builtin.unrealized_conversion_cast %arg8 : bf16 to bf16
      %temp9 = builtin.unrealized_conversion_cast %arg9 : f16 to f16
      %cb0 = d2m.get_cb(0) operand_index = 0 : <memref<32x32xf32, #l1>>
      %cb1 = d2m.get_cb(1) operand_index = 1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %0 = d2m.reserve %cb0 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
      d2m.push %cb0 : <memref<32x32xf32, #l1>>
      %1 = d2m.wait %cb0 : <memref<32x32xf32, #l1>> -> memref<32x32xf32, #l1>
      %2 = d2m.reserve %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      %3 = "d2m.tile_tilize_block"(%1, %2) : (memref<32x32xf32, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1>) -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      d2m.pop %cb0 : <memref<32x32xf32, #l1>>
      d2m.push %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %4 = d2m.wait %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
      d2m.pop %cb1 : <memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
    }
    return
  }
}

// -----

// Test for capturing buffer address arguments.
#l1 = #ttcore.memory_space<l1>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  func.func @add(%arg0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1>, %arg1: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1>) -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1> {
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1>
    %view = d2m.view_layout %arg0 remapping = #map : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%view, %arg1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1>)
        outs(%alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1>)
     {
      %0 = d2m.get_cb(0) : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %1 = d2m.get_cb(1) : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %2 = d2m.get_cb(2) : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %3 = d2m.wait %1 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      %c0 = arith.constant 0 : index
      // CHECK: %4 = d2m.get_arg operand_index = 0 resolution_stage = compile : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1>
      %tx = d2m.dma_read %view[%c0, %c0, %c0], %3[%c0], <1> : (memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> !d2m.mem_tx
      d2m.dma_wait %tx
    }, {
      %0 = d2m.get_cb(0) : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %1 = d2m.get_cb(1) : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %2 = d2m.get_cb(2) : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
    }
    return %alloc : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<0x0, 1>, #l1>
  }
}
