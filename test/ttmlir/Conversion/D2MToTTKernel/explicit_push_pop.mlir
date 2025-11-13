// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttmetal-me-pipeline --convert-d2m-to-ttkernel --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #ttcore.memory_space<l1>
#map_ = affine_map<(d0, d1) -> (d0, d1)>
#parallel_ = #ttcore.iterator_type<parallel>

!ttype_f32 = !ttcore.tile<32x32, f32>

module {
  // Test explicit push/pop for producer/consumer pattern
  // CHECK-LABEL: func.func private @compute_kernel
  func.func @test_explicit_push_pop(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                    %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      // Consumer pattern with explicit pop
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb0 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        %0 = "d2m.tile_abs"(%arg0) : (!ttype_f32) -> !ttype_f32
        linalg.yield %0 : !ttype_f32
      }
      // CHECK: ttkernel.cb_wait_front
      // CHECK: ttkernel.cb_pop_front
      d2m.pop %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>>

      // Producer pattern with explicit push
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        %0 = "d2m.tile_abs"(%arg0) : (!ttype_f32) -> !ttype_f32
        linalg.yield %0 : !ttype_f32
      }
      // CHECK: ttkernel.cb_reserve_back
      // CHECK: ttkernel.cb_push_back
      d2m.push %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>>
    }
    return
  }

}
