// RUN: ttmlir-opt --ttcore-register-device --d2m-schedule-dma -o %t %s
// RUN: FileCheck %s --input-file=%t

#dst = #ttcore.memory_space<dst>
#l1 = #ttcore.memory_space<l1>
#identity = affine_map<() -> ()>

module {
  // Reduced from test_matmul_ttnn_shapes_single_buffered[ttmetal-l1_acc-matmul_tile-f32-2048x2048x2048].
  // The matmul data movement region has two multicast remote loads feeding
  // distinct CBs. Legacy scheduling should split them across the two available
  // datamovement threads and assign the existing noc0/noc1 pairing.
  // CHECK-LABEL: func.func @matmul_two_mcast_remote_loads
  func.func @matmul_two_mcast_remote_loads(
      %arg0: memref<8x64x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>,
      %arg1: memref<64x8x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>) {
    %output = memref.alloc() {alignment = 16 : i64} : memref<8x8x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>
    %lhs_cb_storage = memref.alloc() {alignment = 16 : i64, d2m.cb_for_operand = 0 : i64} : memref<8x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
    %rhs_cb_storage = memref.alloc() {alignment = 16 : i64, d2m.cb_for_operand = 1 : i64} : memref<1x8x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<32768x4096, 2>, #l1>
    %sem0 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %sem1 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %sem2 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore
    %sem3 = d2m.create_local_semaphore <{initialValue = 0 : ui32}> -> !d2m.local_semaphore

    // CHECK: d2m.generic
    // CHECK-SAME: threads = [#d2m.thread<datamovement, noc = 0>, #d2m.thread<datamovement, noc = 1>, #d2m.thread<compute>]
    // CHECK: {
    // CHECK: %[[LHS_CB:.*]] = d2m.get_cb(0) operand_index = 0
    // CHECK-NOT: d2m.get_cb(1)
    // CHECK: scf.for
    // CHECK-NOT: d2m.remote_load %arg1
    // CHECK: d2m.remote_load %arg0{{.*}}into %[[LHS_CB]] mcore{{.*}} {preallocated_semaphores = [3, 4]}
    // CHECK-NOT: d2m.remote_load %arg1
    // CHECK: }, {
    // CHECK: %[[RHS_CB:.*]] = d2m.get_cb(1) operand_index = 1
    // CHECK-NOT: d2m.get_cb(0)
    // CHECK: scf.for
    // CHECK-NOT: d2m.remote_load %arg0
    // CHECK: d2m.remote_load %arg1{{.*}}into %[[RHS_CB]] mcore{{.*}} {preallocated_semaphores = [5, 6]}
    // CHECK-NOT: d2m.remote_load %arg0
    // CHECK: }, {
    // CHECK: %[[OUT_CB:.*]] = d2m.get_cb(2) operand_index = 2
    // CHECK: %[[COMPUTE_RHS_CB:.*]] = d2m.get_cb(1) operand_index = 1
    // CHECK: %[[COMPUTE_LHS_CB:.*]] = d2m.get_cb(0) operand_index = 0
    // CHECK: scf.for
    // CHECK: d2m.wait %[[COMPUTE_LHS_CB]]
    // CHECK: d2m.wait %[[COMPUTE_RHS_CB]]
    // CHECK: d2m.reserve %[[OUT_CB]]
    // CHECK: "d2m.tile_matmul"
    // CHECK: d2m.push %[[OUT_CB]]
    // CHECK: d2m.pop %[[COMPUTE_LHS_CB]]
    // CHECK: d2m.pop %[[COMPUTE_RHS_CB]]
    // CHECK-NOT: d2m.remote_load
    // CHECK: }
    d2m.generic {block_factors = [], grid = #ttcore.grid<8x8>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement>, #d2m.thread<compute>]}
        ins(%arg0, %arg1 : memref<8x64x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>,
                           memref<64x8x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>)
        outs(%output : memref<8x8x8x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1>)
        additionalArgs(%lhs_cb_storage, %rhs_cb_storage, %sem0, %sem1, %sem2, %sem3 :
                       memref<8x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>,
                       memref<1x8x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<32768x4096, 2>, #l1>,
                       !d2m.local_semaphore, !d2m.local_semaphore, !d2m.local_semaphore, !d2m.local_semaphore) {
    ^datamovement0:
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %rhs_cb = d2m.get_cb(1) operand_index = 1 resolution_stage =  compile : <memref<1x8x!ttcore.tile<32x32, f32>, #l1>>
      %lhs_cb = d2m.get_cb(0) operand_index = 0 resolution_stage =  compile : <memref<8x1x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %k = %c0 to %c64 step %c1 {
        %core0 = d2m.core_index(0) {phys_to_virt_map = #identity} : index
        %core1 = d2m.core_index(1) {phys_to_virt_map = #identity} : index
        %mcast_core0 = d2m.core_index(0) {phys_to_virt_map = #identity} : index
        d2m.remote_load %arg0[%core0, %k] into %lhs_cb mcore[%mcast_core0, %c0] mshape[%c1, %c8] {preallocated_semaphores = [3, 4]} : memref<8x64x8x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1> into !d2m.cb<memref<8x1x!ttcore.tile<32x32, f32>, #l1>>
        %mcast_core1 = d2m.core_index(1) {phys_to_virt_map = #identity} : index
        d2m.remote_load %arg1[%k, %core1] into %rhs_cb mcore[%c0, %mcast_core1] mshape[%c8, %c1] {preallocated_semaphores = [5, 6]} : memref<64x8x1x8x!ttcore.tile<32x32, f32>, #ttcore.shard<32768x4096, 1>, #l1> into !d2m.cb<memref<1x8x!ttcore.tile<32x32, f32>, #l1>>
      } {d2m.blocking_loop = 0 : i64}
    }, {
    ^compute0:
      %c64 = arith.constant 64 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %c4 = arith.constant 4 : index
      %out_cb = d2m.get_cb(2) operand_index = 2 resolution_stage =  compile : <memref<8x8x!ttcore.tile<32x32, f32>, #l1>>
      %rhs_cb = d2m.get_cb(1) operand_index = 1 resolution_stage =  compile : <memref<1x8x!ttcore.tile<32x32, f32>, #l1>>
      %lhs_cb = d2m.get_cb(0) operand_index = 0 resolution_stage =  compile : <memref<8x1x!ttcore.tile<32x32, f32>, #l1>>
      scf.for %k = %c0 to %c64 step %c1 {
        %lhs = d2m.wait %lhs_cb : <memref<8x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<8x1x!ttcore.tile<32x32, f32>, #l1>
        %lhs_flat = memref.collapse_shape %lhs [[0, 1]] : memref<8x1x!ttcore.tile<32x32, f32>, #l1> into memref<8x!ttcore.tile<32x32, f32>, #l1>
        %rhs = d2m.wait %rhs_cb : <memref<1x8x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x8x!ttcore.tile<32x32, f32>, #l1>
        %rhs_flat = memref.collapse_shape %rhs [[0, 1]] : memref<1x8x!ttcore.tile<32x32, f32>, #l1> into memref<8x!ttcore.tile<32x32, f32>, #l1>
        %out = d2m.reserve %out_cb : <memref<8x8x!ttcore.tile<32x32, f32>, #l1>> -> memref<8x8x!ttcore.tile<32x32, f32>, #l1>
        %out_flat = memref.collapse_shape %out [[0, 1]] : memref<8x8x!ttcore.tile<32x32, f32>, #l1> into memref<64x!ttcore.tile<32x32, f32>, #l1>
        %lhs_tile = memref.load %lhs_flat[%c0] : memref<8x!ttcore.tile<32x32, f32>, #l1>
        %rhs_tile = memref.load %rhs_flat[%c0] : memref<8x!ttcore.tile<32x32, f32>, #l1>
        %dst = d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, f32>, #dst>
        %acc = memref.load %dst[%c0] : memref<4x!ttcore.tile<32x32, f32>, #dst>
        %result = "d2m.tile_matmul"(%lhs_tile, %rhs_tile, %acc) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        memref.store %result, %out_flat[%c0] : memref<64x!ttcore.tile<32x32, f32>, #l1>
        d2m.push %out_cb : <memref<8x8x!ttcore.tile<32x32, f32>, #l1>>
        %wait_out = d2m.wait %out_cb : <memref<8x8x!ttcore.tile<32x32, f32>, #l1>> -> memref<8x8x!ttcore.tile<32x32, f32>, #l1>
        d2m.pop %out_cb : <memref<8x8x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %lhs_cb : <memref<8x1x!ttcore.tile<32x32, f32>, #l1>>
        d2m.pop %rhs_cb : <memref<1x8x!ttcore.tile<32x32, f32>, #l1>>
      } {d2m.blocking_loop = 0 : i64}
    }

    return
  }
}
