// RUN: ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel %s | FileCheck %s

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  // CHECK-LABEL: func.func @binary_dest_reuse_dst_index_mismatch
  func.func @binary_dest_reuse_dst_index_mismatch(
      %arg0_: !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1_>>,
      %arg1_: !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1_>>,
      %arg2_: !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1_>>)
      attributes {d2m.thread = #d2m.thread<compute>} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index

    %arg0 = d2m.wait %arg0_ : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x!ttcore.tile<32x32, f32>, #l1_>
    %arg1 = d2m.wait %arg1_ : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x!ttcore.tile<32x32, f32>, #l1_>
    %arg2 = d2m.reserve %arg2_ : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x!ttcore.tile<32x32, f32>, #l1_>

    %dst = d2m.acquire_dst() : memref<4x!ttcore.tile<32x32, f32>, #dst_>

    %lhs = memref.load %arg0[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    %rhs = memref.load %arg1[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>
    %rhs_bcast = "d2m.tile_bcast"(%rhs) <{bcast_type = #d2m<tile_bcast_type row>}> : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    memref.store %rhs_bcast, %dst[%c0] : memref<4x!ttcore.tile<32x32, f32>, #dst_>

    %rhs_dst = memref.load %dst[%c0] : memref<4x!ttcore.tile<32x32, f32>, #dst_>
    %out = "d2m.tile_mul"(%lhs, %rhs_dst) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    memref.store %out, %dst[%c1] : memref<4x!ttcore.tile<32x32, f32>, #dst_>

    %result = memref.load %dst[%c1] : memref<4x!ttcore.tile<32x32, f32>, #dst_>
    memref.store %result, %arg2[%c0] : memref<1x!ttcore.tile<32x32, f32>, #l1_>

    d2m.push %arg2_ : <memref<1x!ttcore.tile<32x32, f32>, #l1_>>
    %out_wait = d2m.wait %arg2_ : !d2m.cb<memref<1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x!ttcore.tile<32x32, f32>, #l1_>
    d2m.pop %arg2_ : <memref<1x!ttcore.tile<32x32, f32>, #l1_>>
    d2m.pop %arg1_ : <memref<1x!ttcore.tile<32x32, f32>, #l1_>>
    d2m.pop %arg0_ : <memref<1x!ttcore.tile<32x32, f32>, #l1_>>

    // CHECK: ttkernel.unary_bcast({{.*}}, {{.*}}, <row>)
    // CHECK: ttkernel.copy_dest_values_init
    // CHECK: ttkernel.copy_dest_values(%[[DST_SRC:.*]], %[[DST_OUT:.*]]) : (index, index) -> ()
    // CHECK: ttkernel.binary_dest_reuse_tiles_init({{.*}}, <mul>, <dest_to_srcb>)
    // CHECK: ttkernel.binary_dest_reuse_tiles({{.*}}, {{.*}}, %[[DST_OUT]], <mul>, <dest_to_srcb>)
    return
  }
}
