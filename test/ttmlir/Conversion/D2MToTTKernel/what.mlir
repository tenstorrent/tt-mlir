// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttmetal-me-pipeline --convert-d2m-to-ttkernel --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #ttcore.memory_space<l1>
#map_ = affine_map<(d0, d1) -> (d0, d1)>
#map1_ = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2_ = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3_ = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel_ = #ttcore.iterator_type<parallel>
#reduction_ = #ttcore.iterator_type<reduction>

!ttype_f32 = !ttcore.tile<32x32, f32>
!ttype_si32 = !ttcore.tile<32x32, si32>
!ttype_bf16 = !ttcore.tile<32x32, bf16>



module{

    func.func @test_logical_not_lowering(%in0: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>,
                                       %out: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_bf16, #l1_>) outs(%cb1 : memref<1x1x!ttype_bf16, #l1_>) {
      ^bb0(%arg0: !ttype_bf16, %arg1: !ttype_bf16):
        // CHECK-NOT: d2m.tile_logical_not
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.logical_not_unary_tile_init
        // CHECK: ttkernel.logical_not_unary_tile
        %0 = "d2m.tile_bitwise_not"(%arg0) : (!ttype_bf16) -> !ttype_bf16
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_bf16
      }
    }
    return
  }
}