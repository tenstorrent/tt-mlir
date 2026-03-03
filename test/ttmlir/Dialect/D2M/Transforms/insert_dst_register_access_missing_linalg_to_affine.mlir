// RUN: ttmlir-opt --ttcore-register-device --d2m-insert-dst-register-access %s -split-input-file -verify-diagnostics

#l1_ = #ttcore.memory_space<l1>

// expected-error@below {{found linalg.generic operations that were not converted to affine loops. Please run --d2m-linalg-to-affine before the --d2m-insert-dst-register-access pass.}}
module {
  func.func @test_missing_linalg_to_affine(%in0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>,
                                           %out0: memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>,
                 indexing_maps = [],
                 iterator_types = [],
                 threads = [#d2m.thread<unified>]}
        ins(%in0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%out0 : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<2048x2048, 1>, #l1_>) {
    ^unified0(%arg0_cb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>,
              %arg1_cb: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1_>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1_>

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
}
