// RUN: ttmlir-opt --split-input-file --capture-additional-args-in-threads %s | FileCheck %s
// Test for capturing additional arguments into threads

// Test for capturing scalar types.
#l1 = #ttcore.memory_space<l1>
module {
  func.func @test(%arg0: ui32) {
    %alloc = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %alloc_0 = memref.alloc() {address = 107808 : i64, alignment = 16 : i64} : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_0 : memref<1x1x32x32xf32, #ttcore.shard<32x4, 1>, #l1>)
        outs(%alloc : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)
        additionalArgs(%arg0 : ui32)
     {
    // CHECK: ^unified0(%cb0: !d2m.cb<memref<32x32xf32, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %scalar0: !d2m.scalar<ui32>):
    ^unified0(%cb0: !d2m.cb<memref<32x32xf32, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>):
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

// Test for capturing global semaphore types.
#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#layout1 = #ttcore.metal_layout<logical_shape = 8x8, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
module {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @generic_with_global_semaphore(%arg0: tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout> {
    %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %1 = d2m.empty() : tensor<8x8x1x1xui32, #layout1>
    %2 = d2m.create_global_semaphore(%1) {value = 0 : ui32} : tensor<8x8x1x1xui32, #layout1> -> !d2m.global_semaphore
    %3 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %stream = "d2m.stream_layout"(%arg0, %3) <{remapping = #map}> : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %4 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %stream_0 = "d2m.stream_layout"(%0, %4) <{remapping = #map}> : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>) -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    %5 = d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%stream : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
        outs(%stream_0 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
        additionalArgs(%2 : !d2m.global_semaphore)
     {
    // CHECK: ^unified0(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %gsem0: !d2m.global_semaphore):
    ^unified0(%cb0: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>, %cb1: !d2m.cb<tensor<1x1x!ttcore.tile<32x32, f32>>>):
      %c1 = arith.constant 1 : index
      d2m.semaphore_wait %2, %c1 : !d2m.global_semaphore
      d2m.yield %stream_0 : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
    d2m.reset_global_semaphore(%2) {value = 0 : ui32} : !d2m.global_semaphore
    return %5 : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout>
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
    // CHECK: ^datamovement0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %sem0: !d2m.local_semaphore, %sem1: !d2m.local_semaphore, %sem2: !d2m.local_semaphore, %sem3: !d2m.local_semaphore):
    ^datamovement0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>):
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
    // CHECK: ^datamovement1(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %sem0: !d2m.local_semaphore, %sem1: !d2m.local_semaphore, %sem2: !d2m.local_semaphore, %sem3: !d2m.local_semaphore):
    ^datamovement1(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>):
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
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>):
    }
    return
  }
}

// -----

// Test for inserting scalar type reinterpret casts.
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
    // CHECK: ^unified0(%cb0: !d2m.cb<memref<32x32xf32, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>, %scalar0: !d2m.scalar<ui32>, %scalar1: !d2m.scalar<ui32>, %scalar2: !d2m.scalar<ui32>, %scalar3: !d2m.scalar<ui32>, %scalar4: !d2m.scalar<ui32>, %scalar5: !d2m.scalar<ui32>, %scalar6: !d2m.scalar<ui32>, %scalar7: !d2m.scalar<ui32>, %scalar8: !d2m.scalar<ui32>, %scalar9: !d2m.scalar<ui32>):
    // CHECK: %0 = d2m.reinterpret_cast %scalar9 : !d2m.scalar<ui32> to !d2m.scalar<f16>
    // CHECK: %1 = d2m.reinterpret_cast %scalar8 : !d2m.scalar<ui32> to !d2m.scalar<bf16>
    // CHECK: %2 = d2m.reinterpret_cast %scalar7 : !d2m.scalar<ui32> to !d2m.scalar<f32>
    // CHECK: %3 = d2m.reinterpret_cast %scalar6 : !d2m.scalar<ui32> to !d2m.scalar<i1>
    // CHECK: %4 = d2m.reinterpret_cast %scalar5 : !d2m.scalar<ui32> to !d2m.scalar<si8>
    // CHECK: %5 = d2m.reinterpret_cast %scalar4 : !d2m.scalar<ui32> to !d2m.scalar<ui8>
    // CHECK: %6 = d2m.reinterpret_cast %scalar3 : !d2m.scalar<ui32> to !d2m.scalar<si16>
    // CHECK: %7 = d2m.reinterpret_cast %scalar2 : !d2m.scalar<ui32> to !d2m.scalar<ui16>
    // CHECK: %8 = d2m.reinterpret_cast %scalar1 : !d2m.scalar<ui32> to !d2m.scalar<si32>
    // CHECK: %9 = d2m.reinterpret_cast %scalar0 : !d2m.scalar<ui32> to !d2m.scalar<i32>
    ^unified0(%cb0: !d2m.cb<memref<32x32xf32, #l1>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>):
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