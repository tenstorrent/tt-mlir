// RUN: not ttmlir-opt --ttcore-register-device --convert-d2m-to-ttkernel -o /dev/null %s 2>&1 | FileCheck %s

// The LLK matmul init is emitted once per compute kernel, so a single kernel
// cannot mix tile matmuls that require different transpose_b init values.

#l1_ = #ttcore.memory_space<l1>
#dst_ = #ttcore.memory_space<dst>

module {
  func.func @mixed_matmul_transpose_b() attributes {d2m.thread = #d2m.thread<compute>} {
    %arg0 = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
    %arg1 = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
    %arg2 = d2m.get_cb(2) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>>
    %in0 = d2m.wait %arg0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %in1 = d2m.wait %arg1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %out = d2m.reserve %arg2 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1_>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %dst = d2m.acquire_dst() : memref<2x!ttcore.tile<32x32, f32>, #dst_>
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %a = affine.load %in0[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %b = affine.load %in1[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    %c = affine.load %dst[%c0] : memref<2x!ttcore.tile<32x32, f32>, #dst_>
    // CHECK: error: failed to legalize operation 'd2m.tile_matmul'
    %r0 = "d2m.tile_matmul"(%a, %b, %c) <{transpose_b = true}> : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    affine.store %r0, %dst[%c0] : memref<2x!ttcore.tile<32x32, f32>, #dst_>
    %c_next = affine.load %dst[%c1] : memref<2x!ttcore.tile<32x32, f32>, #dst_>
    %r1 = "d2m.tile_matmul"(%a, %b, %c_next) <{transpose_b = false}> : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
    affine.store %r1, %dst[%c1] : memref<2x!ttcore.tile<32x32, f32>, #dst_>
    %final = affine.load %dst[%c0] : memref<2x!ttcore.tile<32x32, f32>, #dst_>
    affine.store %final, %out[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #l1_>
    return
  }
}
