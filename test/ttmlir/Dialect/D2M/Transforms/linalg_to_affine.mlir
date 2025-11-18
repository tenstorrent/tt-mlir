// RUN: ttmlir-opt --ttcore-register-device --d2m-linalg-to-affine -o %t %s
// RUN: FileCheck %s --input-file=%t

#l1_ = #ttcore.memory_space<l1>

// CHECK-LABEL: func.func @simple_eltwise
func.func @simple_eltwise(%in0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>,
                          %out0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>) {
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                affine_map<(d0, d1) -> (d0, d1)>],
               iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
               threads = [#d2m.thread<compute>]}
      ins(%in0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>)
      outs(%out0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>) {
  ^compute0(%arg0_cb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>,
            %arg1_cb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

    // CHECK-NOT: linalg.generic
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: affine.load
    // CHECK: d2m.tile_exp
    // CHECK: affine.store
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                     affine_map<(d0, d1) -> (d0, d1)>],
                    iterator_types = ["parallel", "parallel"]}
        ins(%cb0 : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
        outs(%cb1 : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) {
    ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>):
      %0 = "d2m.tile_exp"(%arg0) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %0 : !ttcore.tile<32x32, f32>
    }
  }
  return
}

// CHECK-LABEL: func.func @binary_eltwise
func.func @binary_eltwise(%in0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>,
                          %in1: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>,
                          %out0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>) {
  d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                affine_map<(d0, d1) -> (d0, d1)>,
                                affine_map<(d0, d1) -> (d0, d1)>],
               iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
               threads = [#d2m.thread<compute>]}
      ins(%in0, %in1 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>,
                       memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>)
      outs(%out0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>) {
  ^compute0(%arg0_cb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>,
            %arg1_cb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>,
            %arg2_cb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
    %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

    // CHECK-NOT: linalg.generic
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: affine.load
    // CHECK: affine.load
    // CHECK: d2m.tile_add
    // CHECK: affine.store
    linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                                     affine_map<(d0, d1) -> (d0, d1)>,
                                     affine_map<(d0, d1) -> (d0, d1)>],
                    iterator_types = ["parallel", "parallel"]}
        ins(%cb0, %cb1 : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>,
                         memref<2x4x!ttcore.tile<32x32, f32>, #l1_>)
        outs(%cb2 : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) {
    ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
      %0 = "d2m.tile_add"(%arg0, %arg1) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %0 : !ttcore.tile<32x32, f32>
    }
  }
  return
}

// CHECK-LABEL: func.func @matmul
func.func @matmul(%in0: memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                  %in1: memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                  %out0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
  d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>,
               indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                affine_map<(d0, d1, d2) -> (d2, d1)>,
                                affine_map<(d0, d1, d2) -> (d0, d1)>],
               iterator_types = [#ttcore.iterator_type<parallel>,
                                #ttcore.iterator_type<parallel>,
                                #ttcore.iterator_type<reduction>],
               threads = [#d2m.thread<compute>]}
      ins(%in0, %in1 : memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>,
                       memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>)
      outs(%out0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1_>) {
  ^compute0(%arg0_cb: !d2m.cb<memref<2x3x!ttcore.tile<32x32, f32>, #l1_>>,
            %arg1_cb: !d2m.cb<memref<3x4x!ttcore.tile<32x32, f32>, #l1_>>,
            %arg2_cb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
    %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<2x3x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x3x!ttcore.tile<32x32, f32>, #l1_>
    %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<3x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<3x4x!ttcore.tile<32x32, f32>, #l1_>
    %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

    // CHECK-NOT: linalg.generic
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: affine.for
    // CHECK: affine.load
    // CHECK: affine.load
    // CHECK: affine.load
    // CHECK: d2m.tile_matmul
    // CHECK: affine.store
    linalg.generic {indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>,
                                     affine_map<(d0, d1, d2) -> (d2, d1)>,
                                     affine_map<(d0, d1, d2) -> (d0, d1)>],
                    iterator_types = ["parallel", "parallel", "reduction"]}
        ins(%cb0, %cb1 : memref<2x3x!ttcore.tile<32x32, f32>, #l1_>,
                         memref<3x4x!ttcore.tile<32x32, f32>, #l1_>)
        outs(%cb2 : memref<2x4x!ttcore.tile<32x32, f32>, #l1_>) {
    ^bb0(%arg0: !ttcore.tile<32x32, f32>, %arg1: !ttcore.tile<32x32, f32>, %arg2: !ttcore.tile<32x32, f32>):
      %0 = "d2m.tile_matmul"(%arg0, %arg1, %arg2) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %0 : !ttcore.tile<32x32, f32>
    }
  }
  return
}
