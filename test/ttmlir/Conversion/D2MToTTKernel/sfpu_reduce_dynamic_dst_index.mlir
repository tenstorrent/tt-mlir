// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @sfpu_reduce_dynamic_dst_index
  func.func @sfpu_reduce_dynamic_dst_index() attributes {d2m.thread = #d2m.thread<compute>} {
    %cb0 = d2m.get_cb(0) : !d2m.cb<memref<4x!ttcore.tile<32x32, si32>, #l1_>>
    %cb1 = d2m.get_cb(1) : !d2m.cb<memref<4x!ttcore.tile<32x32, si32>, #l1_>>
    %in_buf = d2m.wait %cb0 : !d2m.cb<memref<4x!ttcore.tile<32x32, si32>, #l1_>> -> memref<4x!ttcore.tile<32x32, si32>, #l1_>
    %out_buf = d2m.reserve %cb1 : !d2m.cb<memref<4x!ttcore.tile<32x32, si32>, #l1_>> -> memref<4x!ttcore.tile<32x32, si32>, #l1_>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dst = d2m.acquire_dst() : memref<5x!ttcore.tile<32x32, si32>, #dst_>
    scf.for %i = %c0 to %c4 step %c1 {
      %a = memref.load %in_buf[%i] : memref<4x!ttcore.tile<32x32, si32>, #l1_>
      %c = memref.load %dst[%i] : memref<5x!ttcore.tile<32x32, si32>, #dst_>
      // CHECK: scf.for %[[I:arg[0-9]+]]
      // CHECK: ttkernel.fill_tile_int(%[[I]],
      // CHECK: ttkernel.sfpu_reduce
      // CHECK: ttkernel.add_int_tile({{.*}}, %[[I]], %[[I]],
      %reduced = "d2m.tile_sfpu_reduce_sum"(%a, %c) <{dst_scratch_index = 4 : i64, reduce_dim = #d2m<reduce_dim C>}> : (!ttcore.tile<32x32, si32>, !ttcore.tile<32x32, si32>) -> !ttcore.tile<32x32, si32>
      memref.store %reduced, %dst[%i] : memref<5x!ttcore.tile<32x32, si32>, #dst_>
      %result = memref.load %dst[%i] : memref<5x!ttcore.tile<32x32, si32>, #dst_>
      memref.store %result, %out_buf[%i] : memref<4x!ttcore.tile<32x32, si32>, #l1_>
    }
    return
  }

  // CHECK-LABEL: func.func @sfpu_reduce_dynamic_dst_index_loop_carried
  func.func @sfpu_reduce_dynamic_dst_index_loop_carried() attributes {d2m.thread = #d2m.thread<compute>} {
    %cb0 = d2m.get_cb(0) : !d2m.cb<memref<4x!ttcore.tile<32x32, si32>, #l1_>>
    %cb1 = d2m.get_cb(1) : !d2m.cb<memref<4x!ttcore.tile<32x32, si32>, #l1_>>
    %in_buf = d2m.wait %cb0 : !d2m.cb<memref<4x!ttcore.tile<32x32, si32>, #l1_>> -> memref<4x!ttcore.tile<32x32, si32>, #l1_>
    %out_buf = d2m.reserve %cb1 : !d2m.cb<memref<4x!ttcore.tile<32x32, si32>, #l1_>> -> memref<4x!ttcore.tile<32x32, si32>, #l1_>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %dst = d2m.acquire_dst() : memref<5x!ttcore.tile<32x32, si32>, #dst_>
    scf.for %i = %c0 to %c4 step %c1 {
      scf.for %k = %c0 to %c4 step %c1 {
        %a = memref.load %in_buf[%k] : memref<4x!ttcore.tile<32x32, si32>, #l1_>
        %c = memref.load %dst[%i] : memref<5x!ttcore.tile<32x32, si32>, #dst_>
        // CHECK: scf.for %[[I:arg[0-9]+]]
        // CHECK: scf.for %[[K:arg[0-9]+]]
        // CHECK: %[[FIRST:.*]] = arith.cmpi eq, %[[K]], %{{.*}} : index
        // CHECK: scf.if %[[FIRST]] {
        // CHECK: ttkernel.fill_tile_int(%[[I]],
        // CHECK: }
        // CHECK: ttkernel.sfpu_reduce
        // CHECK: ttkernel.add_int_tile({{.*}}, %[[I]], %[[I]],
        %reduced = "d2m.tile_sfpu_reduce_sum"(%a, %c) <{dst_scratch_index = 4 : i64, reduce_dim = #d2m<reduce_dim C>}> : (!ttcore.tile<32x32, si32>, !ttcore.tile<32x32, si32>) -> !ttcore.tile<32x32, si32>
        memref.store %reduced, %dst[%i] : memref<5x!ttcore.tile<32x32, si32>, #dst_>
      }
      %result = memref.load %dst[%i] : memref<5x!ttcore.tile<32x32, si32>, #dst_>
      memref.store %result, %out_buf[%i] : memref<4x!ttcore.tile<32x32, si32>, #l1_>
    }
    return
  }
}
