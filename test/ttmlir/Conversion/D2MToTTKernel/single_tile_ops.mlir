// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttmetal-me-pipeline="dst-allocation-strategy=legacy" --convert-d2m-to-ttkernel --canonicalize %s | FileCheck %s --check-prefix=COMMON
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttmetal-me-pipeline="dst-allocation-strategy=graph-coloring-greedy" --convert-d2m-to-ttkernel --canonicalize %s | FileCheck %s --check-prefix=COMMON
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttmetal-me-pipeline="dst-allocation-strategy=graph-coloring-cb" --convert-d2m-to-ttkernel --canonicalize %s | FileCheck %s --check-prefix=COMMON

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

module {
  //===----------------------------------------------------------------------===//
  // TTIR FPU operations
  //===----------------------------------------------------------------------===//

  func.func @test_matmul_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                  %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                  %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map1_, #map2_, #map3_], iterator_types = [#parallel_, #parallel_, #reduction_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map1_, #map2_, #map3_], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // COMMON-NOT: d2m.tile_matmul
        // COMMON: ttkernel.mm_block_init
        // COMMON: ttkernel.mm_block_init_short
        // COMMON: ttkernel.experimental::matmul_block
        %0 = "d2m.tile_matmul"(%arg0, %arg1, %arg2) : (!ttype_f32, !ttype_f32, !ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_add_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // COMMON-NOT: d2m.tile_add
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.add_binary_tile_init
        // COMMON: ttkernel.add_binary_tile
        %0 = "d2m.tile_add"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_sub_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // COMMON-NOT: d2m.tile_sub
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.sub_binary_tile_init
        // COMMON: ttkernel.sub_binary_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_mul_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // COMMON-NOT: d2m.tile_mul
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.mul_binary_tile_init
        // COMMON: ttkernel.mul_binary_tile
        %0 = "d2m.tile_mul"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  //===----------------------------------------------------------------------===//
  // TTIR SFPU operations
  //===----------------------------------------------------------------------===//

  func.func @test_max_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // COMMON-NOT: d2m.tile_max
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]]) :
        // COMMON: ttkernel.copy_tile_init(%[[CB1:.+]]) :
        // COMMON-NOT: ttkernel.copy_tile(%{{.+}}, %{{.+}}, %[[DST_IDX0]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB1]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.max_tile_init
        // COMMON: ttkernel.max_tile(%{{.+}}, %{{.+}})
        %0 = "d2m.tile_maximum"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_div_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // COMMON-NOT: d2m.tile_div
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]]) :
        // COMMON: ttkernel.copy_tile_init(%[[CB1:.+]]) :
        // COMMON-NOT: ttkernel.copy_tile(%{{.+}}, %{{.+}}, %[[DST_IDX0]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB1]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.div_binary_tile_init
        // COMMON: ttkernel.div_binary_tile(%{{.+}}, %{{.+}}, %{{.+}})
        %0 = "d2m.tile_div"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_recip_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                 %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_recip
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.recip_tile_init
        // COMMON: ttkernel.recip_tile
        %0 = "d2m.tile_recip"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_pow_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // COMMON-NOT: d2m.tile_pow
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]]) :
        // COMMON: ttkernel.copy_tile_init(%[[CB1:.+]]) :
        // COMMON-NOT: ttkernel.copy_tile(%{{.+}}, %{{.+}}, %[[DST_IDX0]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB1]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.power_binary_tile_init
        // COMMON: ttkernel.power_binary_tile(%{{.+}}, %{{.+}}, %{{.+}})
        %0 = "d2m.tile_pow"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_exp_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_exp
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.exp_tile_init
        // COMMON: ttkernel.exp_tile
        %0 = "d2m.tile_exp"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_log_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_log
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.log_tile_init
        // COMMON: ttkernel.log_tile
        %0 = "d2m.tile_log"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_cos_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_cos
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.cos_tile_init
        // COMMON: ttkernel.cos_tile
        %0 = "d2m.tile_cos"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_tan_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_tan
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.tan_tile_init
        // COMMON: ttkernel.tan_tile
        %0 = "d2m.tile_tan"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_negative_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                    %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_neg
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.negative_tile_init
        // COMMON: ttkernel.negative_tile
        %0 = "d2m.tile_negative"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_sqrt_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_sqrt
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.sqrt_tile_init
        // COMMON: ttkernel.sqrt_tile
        %0 = "d2m.tile_sqrt"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_rsqrt_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                 %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_rsqrt
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.rsqrt_tile_init
        // COMMON: ttkernel.rsqrt_tile
        %0 = "d2m.tile_rsqrt"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_sin_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_sin
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.sin_tile_init
        // COMMON: ttkernel.sin_tile
        %0 = "d2m.tile_sin"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_sigmoid_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                   %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_sigmoid
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.sigmoid_tile_init
        // COMMON: ttkernel.sigmoid_tile
        %0 = "d2m.tile_sigmoid"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_gelu_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_gelu
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.gelu_tile_init
        // COMMON: ttkernel.gelu_tile
        %0 = "d2m.tile_gelu"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_ceil_lowering(%in0: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>,
                                %out: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_bf16, #l1_>) outs(%cb1 : memref<1x1x!ttype_bf16, #l1_>) {
      ^bb0(%arg0: !ttype_bf16, %arg1: !ttype_bf16):
        // COMMON-NOT: d2m.tile_ceil
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.rounding_op_tile_init
        // COMMON: ttkernel.ceil_tile
        %0 = "d2m.tile_ceil"(%arg0) : (!ttype_bf16) -> !ttype_bf16
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_bf16
      }
    }
    return
  }

  func.func @test_ceil_lowering_f32(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                    %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_ceil
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.rounding_op_tile_init
        // COMMON: ttkernel.ceil_tile_float32
        %0 = "d2m.tile_ceil"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_floor_lowering(%in0: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>,
                                 %out: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_bf16, #l1_>) outs(%cb1 : memref<1x1x!ttype_bf16, #l1_>) {
      ^bb0(%arg0: !ttype_bf16, %arg1: !ttype_bf16):
        // COMMON-NOT: d2m.tile_floor
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.rounding_op_tile_init
        // COMMON: ttkernel.floor_tile
        %0 = "d2m.tile_floor"(%arg0) : (!ttype_bf16) -> !ttype_bf16
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_bf16
      }
    }
    return
  }

  func.func @test_floor_lowering_f32(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                     %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_floor
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.rounding_op_tile_init
        // COMMON: ttkernel.floor_tile_float32
        %0 = "d2m.tile_floor"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_abs_lowering(%in0: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_bf16, #l1_>) outs(%cb1 : memref<1x1x!ttype_bf16, #l1_>) {
      ^bb0(%arg0: !ttype_bf16, %arg1: !ttype_bf16):
        // COMMON-NOT: d2m.tile_abs
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.abs_tile_init
        // COMMON: ttkernel.abs_tile
        %0 = "d2m.tile_abs"(%arg0) : (!ttype_bf16) -> !ttype_bf16
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_bf16
      }
    }
    return
  }

  func.func @test_abs_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                   %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_si32, #l1_>) outs(%cb1 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32):
        // COMMON-NOT: d2m.tile_abs
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.abs_tile_init
        // COMMON: ttkernel.abs_tile_int32
        %0 = "d2m.tile_abs"(%arg0) : (!ttype_si32) -> !ttype_si32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_si32
      }
    }
    return
  }

  func.func @test_bitwise_not_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<2048x2048, 1>, #l1_>,
                                       %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<2048x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<2048x2048, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_si32, #l1_>) outs(%cb1 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32):
        // COMMON-NOT: d2m.tile_bitwise_not
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.bitwise_not_tile_init
        // COMMON: ttkernel.bitwise_not_tile
        %0 = "d2m.tile_bitwise_not"(%arg0) : (!ttype_si32) -> !ttype_si32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_si32
      }
    }
    return
  }

  func.func @test_logical_not_lowering(%in0: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>,
                                       %out: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_bf16, #l1_>) outs(%cb1 : memref<1x1x!ttype_bf16, #l1_>) {
      ^bb0(%arg0: !ttype_bf16, %arg1: !ttype_bf16):
        // COMMON-NOT: d2m.tile_logical_not
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.logical_not_unary_tile_init
        // COMMON: ttkernel.logical_not_unary_tile
        %0 = "d2m.tile_logical_not"(%arg0) : (!ttype_bf16) -> !ttype_bf16
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_bf16
      }
    }
    return
  }

  func.func @test_logical_not_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                           %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_si32, #l1_>) outs(%cb1 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32):
        // COMMON-NOT: d2m.tile_logical_not
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.logical_not_unary_tile_init
        // COMMON: ttkernel.logical_not_unary_tile_int32
        %0 = "d2m.tile_logical_not"(%arg0) : (!ttype_si32) -> !ttype_si32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_si32
      }
    }
    return
  }

  //===----------------------------------------------------------------------===//
  // TTIR Comparison operations
  //===----------------------------------------------------------------------===//

  func.func @test_equal_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                 %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                 %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // COMMON-NOT: d2m.tile_sub
        // COMMON-NOT: d2m.tile_eqz
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.sub_binary_tile_init
        // COMMON: ttkernel.sub_binary_tile
        // COMMON: ttkernel.eqz_tile_init
        // COMMON: ttkernel.eqz_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        %1 = "d2m.tile_eqz"(%0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %1 : !ttype_f32
      }
    }
    return
  }

  func.func @test_not_equal_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                     %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                     %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // COMMON-NOT: d2m.tile_sub
        // COMMON-NOT: d2m.tile_nez
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.sub_binary_tile_init
        // COMMON: ttkernel.sub_binary_tile
        // COMMON: ttkernel.nez_tile_init
        // COMMON: ttkernel.nez_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        %1 = "d2m.tile_nez"(%0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %1 : !ttype_f32
      }
    }
    return
  }

  func.func @test_greater_than_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                        %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                        %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // COMMON-NOT: d2m.tile_sub
        // COMMON-NOT: d2m.tile_gtz
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.sub_binary_tile_init
        // COMMON: ttkernel.sub_binary_tile
        // COMMON: ttkernel.gtz_tile_init
        // COMMON: ttkernel.gtz_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        %1 = "d2m.tile_gtz"(%0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %1 : !ttype_f32
      }
    }
    return
  }

  func.func @test_greater_equal_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                         %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                         %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // COMMON-NOT: d2m.tile_sub
        // COMMON-NOT: d2m.tile_gez
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.sub_binary_tile_init
        // COMMON: ttkernel.sub_binary_tile
        // COMMON: ttkernel.gez_tile_init
        // COMMON: ttkernel.gez_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        %1 = "d2m.tile_gez"(%0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %1 : !ttype_f32
      }
    }
    return
  }

  func.func @test_less_than_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                     %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                     %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // COMMON-NOT: d2m.tile_sub
        // COMMON-NOT: d2m.tile_ltz
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.sub_binary_tile_init
        // COMMON: ttkernel.sub_binary_tile
        // COMMON: ttkernel.ltz_tile_init
        // COMMON: ttkernel.ltz_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        %1 = "d2m.tile_ltz"(%0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %1 : !ttype_f32
      }
    }
    return
  }

  func.func @test_less_equal_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                      %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                      %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // COMMON-NOT: d2m.tile_sub
        // COMMON-NOT: d2m.tile_lez
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.sub_binary_tile_init
        // COMMON: ttkernel.sub_binary_tile
        // COMMON: ttkernel.lez_tile_init
        // COMMON: ttkernel.lez_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        %1 = "d2m.tile_lez"(%0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %1 : !ttype_f32
      }
    }
    return
  }

  //===----------------------------------------------------------------------===//
  // TTIR Comparison operations (i32)
  //===----------------------------------------------------------------------===//

  func.func @test_equal_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                     %in1: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                     %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_si32, #l1_>, memref<1x1x!ttype_si32, #l1_>) outs(%cb2 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32, %arg2: !ttype_si32):
        // COMMON-NOT: d2m.tile_sub
        // COMMON-NOT: d2m.tile_eqz
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.sub_binary_tile_init
        // COMMON: ttkernel.sub_binary_tile
        // COMMON: ttkernel.eqz_tile_init
        // COMMON: ttkernel.eqz_tile_int32
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_si32, !ttype_si32) -> !ttype_si32
        %1 = "d2m.tile_eqz"(%0) : (!ttype_si32) -> !ttype_si32
        // COMMON: ttkernel.pack_tile
        linalg.yield %1 : !ttype_si32
      }
    }
    return
  }

  func.func @test_not_equal_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                         %in1: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                         %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_si32, #l1_>, memref<1x1x!ttype_si32, #l1_>) outs(%cb2 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32, %arg2: !ttype_si32):
        // COMMON-NOT: d2m.tile_sub
        // COMMON-NOT: d2m.tile_nez
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.sub_binary_tile_init
        // COMMON: ttkernel.sub_binary_tile
        // COMMON: ttkernel.nez_tile_init
        // COMMON: ttkernel.nez_tile_int32
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_si32, !ttype_si32) -> !ttype_si32
        %1 = "d2m.tile_nez"(%0) : (!ttype_si32) -> !ttype_si32
        // COMMON: ttkernel.pack_tile
        linalg.yield %1 : !ttype_si32
      }
    }
    return
  }

  func.func @test_greater_than_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                            %in1: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                            %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_si32, #l1_>, memref<1x1x!ttype_si32, #l1_>) outs(%cb2 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32, %arg2: !ttype_si32):
        // COMMON-NOT: d2m.tile_sub
        // COMMON-NOT: d2m.tile_gtz
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.sub_binary_tile_init
        // COMMON: ttkernel.sub_binary_tile
        // COMMON: ttkernel.gtz_tile_init
        // COMMON: ttkernel.gtz_tile_int32
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_si32, !ttype_si32) -> !ttype_si32
        %1 = "d2m.tile_gtz"(%0) : (!ttype_si32) -> !ttype_si32
        // COMMON: ttkernel.pack_tile
        linalg.yield %1 : !ttype_si32
      }
    }
    return
  }

  func.func @test_greater_equal_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                             %in1: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                             %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_si32, #l1_>, memref<1x1x!ttype_si32, #l1_>) outs(%cb2 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32, %arg2: !ttype_si32):
        // COMMON-NOT: d2m.tile_sub
        // COMMON-NOT: d2m.tile_gez
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.sub_binary_tile_init
        // COMMON: ttkernel.sub_binary_tile
        // COMMON: ttkernel.gez_tile_init
        // COMMON: ttkernel.gez_tile_int32
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_si32, !ttype_si32) -> !ttype_si32
        %1 = "d2m.tile_gez"(%0) : (!ttype_si32) -> !ttype_si32
        // COMMON: ttkernel.pack_tile
        linalg.yield %1 : !ttype_si32
      }
    }
    return
  }

  func.func @test_less_than_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                         %in1: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                         %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_si32, #l1_>, memref<1x1x!ttype_si32, #l1_>) outs(%cb2 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32, %arg2: !ttype_si32):
        // COMMON-NOT: d2m.tile_sub
        // COMMON-NOT: d2m.tile_ltz
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.sub_binary_tile_init
        // COMMON: ttkernel.sub_binary_tile
        // COMMON: ttkernel.ltz_tile_init
        // COMMON: ttkernel.ltz_tile_int32
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_si32, !ttype_si32) -> !ttype_si32
        %1 = "d2m.tile_ltz"(%0) : (!ttype_si32) -> !ttype_si32
        // COMMON: ttkernel.pack_tile
        linalg.yield %1 : !ttype_si32
      }
    }
    return
  }

  func.func @test_less_equal_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                          %in1: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                          %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>, memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_si32, #l1_>, memref<1x1x!ttype_si32, #l1_>) outs(%cb2 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32, %arg2: !ttype_si32):
        // COMMON-NOT: d2m.tile_sub
        // COMMON-NOT: d2m.tile_lez
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.sub_binary_tile_init
        // COMMON: ttkernel.sub_binary_tile
        // COMMON: ttkernel.lez_tile_init
        // COMMON: ttkernel.lez_tile_int32
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_si32, !ttype_si32) -> !ttype_si32
        %1 = "d2m.tile_lez"(%0) : (!ttype_si32) -> !ttype_si32
        // COMMON: ttkernel.pack_tile
        linalg.yield %1 : !ttype_si32
      }
    }
    return
  }

  func.func @test_silu_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_silu
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.silu_tile_init
        // COMMON: ttkernel.silu_tile
        %0 = "d2m.tile_silu"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_relu_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // COMMON-NOT: d2m.tile_relu
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.relu_tile_init
        // COMMON: ttkernel.relu_tile
        %0 = "d2m.tile_relu"(%arg0) : (!ttype_f32) -> !ttype_f32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_relu_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                   %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_si32, #l1_>) outs(%cb1 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32):
        // COMMON-NOT: d2m.tile_relu
        // COMMON: ttkernel.init_sfpu
        // COMMON: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // COMMON-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // COMMON: ttkernel.relu_tile_init
        // COMMON: ttkernel.relu_tile_int32
        %0 = "d2m.tile_relu"(%arg0) : (!ttype_si32) -> !ttype_si32
        // COMMON: ttkernel.pack_tile
        linalg.yield %0 : !ttype_si32
      }
    }
    return
  }
}
