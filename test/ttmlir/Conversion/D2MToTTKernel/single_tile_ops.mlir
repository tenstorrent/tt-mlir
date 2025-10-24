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

module {
  //===----------------------------------------------------------------------===//
  // TTIR FPU operations
  //===----------------------------------------------------------------------===//

  func.func @test_matmul_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                  %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                  %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map1_, #map2_, #map3_], iterator_types = [#parallel_, #parallel_, #reduction_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map1_, #map2_, #map3_], iterator_types = ["parallel", "parallel", "reduction"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // CHECK-NOT: d2m.tile_matmul
        // CHECK: ttkernel.mm_block_init
        // CHECK: ttkernel.mm_block_init_short
        // CHECK: ttkernel.experimental::matmul_block
        %0 = "d2m.tile_matmul"(%arg0, %arg1, %arg2) : (!ttype_f32, !ttype_f32, !ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_add_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // CHECK-NOT: d2m.tile_add
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.add_binary_tile_init
        // CHECK: ttkernel.add_binary_tile
        %0 = "d2m.tile_add"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_sub_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // CHECK-NOT: d2m.tile_sub
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.sub_binary_tile_init
        // CHECK: ttkernel.sub_binary_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_mul_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // CHECK-NOT: d2m.tile_mul
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.mul_binary_tile_init
        // CHECK: ttkernel.mul_binary_tile
        %0 = "d2m.tile_mul"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  //===----------------------------------------------------------------------===//
  // TTIR SFPU operations
  //===----------------------------------------------------------------------===//

  func.func @test_max_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // CHECK-NOT: d2m.tile_max
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]]) :
        // CHECK: ttkernel.copy_tile_init(%[[CB1:.+]]) :
        // CHECK-NOT: ttkernel.copy_tile(%{{.+}}, %{{.+}}, %[[DST_IDX0]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB1]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.max_tile_init
        // CHECK: ttkernel.max_tile(%{{.+}}, %{{.+}})
        %0 = "d2m.tile_maximum"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_div_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // CHECK-NOT: d2m.tile_div
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]]) :
        // CHECK: ttkernel.copy_tile_init(%[[CB1:.+]]) :
        // CHECK-NOT: ttkernel.copy_tile(%{{.+}}, %{{.+}}, %[[DST_IDX0]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB1]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.div_binary_tile_init
        // CHECK: ttkernel.div_binary_tile(%{{.+}}, %{{.+}}, %{{.+}})
        %0 = "d2m.tile_div"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_recip_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                 %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // CHECK-NOT: d2m.tile_recip
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.recip_tile_init
        // CHECK: ttkernel.recip_tile
        %0 = "d2m.tile_recip"(%arg0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_pow_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // CHECK-NOT: d2m.tile_pow
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %[[DST_IDX0:.+]]) :
        // CHECK: ttkernel.copy_tile_init(%[[CB1:.+]]) :
        // CHECK-NOT: ttkernel.copy_tile(%{{.+}}, %{{.+}}, %[[DST_IDX0]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB1]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.power_binary_tile_init
        // CHECK: ttkernel.power_binary_tile(%{{.+}}, %{{.+}}, %{{.+}})
        %0 = "d2m.tile_pow"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_exp_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // CHECK-NOT: d2m.tile_exp
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.exp_tile_init
        // CHECK: ttkernel.exp_tile
        %0 = "d2m.tile_exp"(%arg0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_log_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // CHECK-NOT: d2m.tile_log
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.log_tile_init
        // CHECK: ttkernel.log_tile
        %0 = "d2m.tile_log"(%arg0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_cos_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // CHECK-NOT: d2m.tile_cos
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.cos_tile_init
        // CHECK: ttkernel.cos_tile
        %0 = "d2m.tile_cos"(%arg0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_tan_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // CHECK-NOT: d2m.tile_tan
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.tan_tile_init
        // CHECK: ttkernel.tan_tile
        %0 = "d2m.tile_tan"(%arg0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_negative_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                    %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // CHECK-NOT: d2m.tile_neg
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.negative_tile_init
        // CHECK: ttkernel.negative_tile
        %0 = "d2m.tile_negative"(%arg0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_sqrt_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // CHECK-NOT: d2m.tile_sqrt
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.sqrt_tile_init
        // CHECK: ttkernel.sqrt_tile
        %0 = "d2m.tile_sqrt"(%arg0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_rsqrt_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                 %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // CHECK-NOT: d2m.tile_rsqrt
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.rsqrt_tile_init
        // CHECK: ttkernel.rsqrt_tile
        %0 = "d2m.tile_rsqrt"(%arg0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_sin_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // CHECK-NOT: d2m.tile_sin
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.sin_tile_init
        // CHECK: ttkernel.sin_tile
        %0 = "d2m.tile_sin"(%arg0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_sigmoid_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                   %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // CHECK-NOT: d2m.tile_sigmoid
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.sigmoid_tile_init
        // CHECK: ttkernel.sigmoid_tile
        %0 = "d2m.tile_sigmoid"(%arg0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_gelu_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // CHECK-NOT: d2m.tile_gelu
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.gelu_tile_init
        // CHECK: ttkernel.gelu_tile
        %0 = "d2m.tile_gelu"(%arg0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_ceil_lowering(%in0: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>,
                                %out: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_bf16, #l1_>) outs(%cb1 : memref<1x1x!ttype_bf16, #l1_>) {
      ^bb0(%arg0: !ttype_bf16, %arg1: !ttype_bf16):
        // CHECK-NOT: d2m.tile_ceil
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.rounding_op_tile_init
        // CHECK: ttkernel.ceil_tile
        %0 = "d2m.tile_ceil"(%arg0) : (!ttype_bf16) -> !ttype_bf16
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_bf16
      }
    }
    return
  }

  func.func @test_ceil_lowering_f32(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                    %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // CHECK-NOT: d2m.tile_ceil
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.rounding_op_tile_init
        // CHECK: ttkernel.ceil_tile_float32
        %0 = "d2m.tile_ceil"(%arg0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_floor_lowering(%in0: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>,
                                 %out: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_bf16, #l1_>) outs(%cb1 : memref<1x1x!ttype_bf16, #l1_>) {
      ^bb0(%arg0: !ttype_bf16, %arg1: !ttype_bf16):
        // CHECK-NOT: d2m.tile_floor
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.rounding_op_tile_init
        // CHECK: ttkernel.floor_tile
        %0 = "d2m.tile_floor"(%arg0) : (!ttype_bf16) -> !ttype_bf16
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_bf16
      }
    }
    return
  }

  func.func @test_floor_lowering_f32(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                     %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        // CHECK-NOT: d2m.tile_floor
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.rounding_op_tile_init
        // CHECK: ttkernel.floor_tile_float32
        %0 = "d2m.tile_floor"(%arg0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_abs_lowering(%in0: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>,
                               %out: memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_bf16, #ttcore.shard<2048x2048>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_bf16, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_bf16, #l1_>> -> memref<1x1x!ttype_bf16, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_bf16, #l1_>) outs(%cb1 : memref<1x1x!ttype_bf16, #l1_>) {
      ^bb0(%arg0: !ttype_bf16, %arg1: !ttype_bf16):
        // CHECK-NOT: d2m.tile_abs
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.abs_tile_init
        // CHECK: ttkernel.abs_tile
        %0 = "d2m.tile_abs"(%arg0) : (!ttype_bf16) -> !ttype_bf16
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_bf16
      }
    }
    return
  }

  func.func @test_abs_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                   %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_si32, #l1_>) outs(%cb1 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32):
        // CHECK-NOT: d2m.tile_abs
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.abs_tile_init
        // CHECK: ttkernel.abs_tile_int32
        %0 = "d2m.tile_abs"(%arg0) : (!ttype_si32) -> !ttype_si32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_si32
      }
    }
    return
  }

  func.func @test_bitwise_not_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<2048x2048>, #l1_>,
                                       %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<2048x2048>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<2048x2048>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<2048x2048>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_si32, #l1_>) outs(%cb1 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32):
        // CHECK-NOT: d2m.tile_bitwise_not
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.bitwise_not_tile_init
        // CHECK: ttkernel.bitwise_not_tile
        %0 = "d2m.tile_bitwise_not"(%arg0) : (!ttype_si32) -> !ttype_si32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_si32
      }
    }
    return
  }

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
        %0 = "d2m.tile_logical_not"(%arg0) : (!ttype_bf16) -> !ttype_bf16
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_bf16
      }
    }
    return
  }

  func.func @test_logical_not_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                           %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_si32, #l1_>) outs(%cb1 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32):
        // CHECK-NOT: d2m.tile_logical_not
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.copy_tile_init(%[[CB0:.+]]) :
        // CHECK-NEXT: ttkernel.copy_tile(%[[CB0]], %{{.+}}, %{{.+}}) :
        // CHECK: ttkernel.logical_not_unary_tile_init
        // CHECK: ttkernel.logical_not_unary_tile_int32
        %0 = "d2m.tile_logical_not"(%arg0) : (!ttype_si32) -> !ttype_si32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_si32
      }
    }
    return
  }

  //===----------------------------------------------------------------------===//
  // TTIR Comparison operations
  //===----------------------------------------------------------------------===//

  func.func @test_equal_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                 %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                 %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // CHECK-NOT: d2m.tile_sub
        // CHECK-NOT: d2m.tile_eqz
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.sub_binary_tile_init
        // CHECK: ttkernel.sub_binary_tile
        // CHECK: ttkernel.eqz_tile_init
        // CHECK: ttkernel.eqz_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        %1 = "d2m.tile_eqz"(%0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %1 : !ttype_f32
      }
    }
    return
  }

  func.func @test_not_equal_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                     %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                     %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // CHECK-NOT: d2m.tile_sub
        // CHECK-NOT: d2m.tile_nez
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.sub_binary_tile_init
        // CHECK: ttkernel.sub_binary_tile
        // CHECK: ttkernel.nez_tile_init
        // CHECK: ttkernel.nez_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        %1 = "d2m.tile_nez"(%0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %1 : !ttype_f32
      }
    }
    return
  }

  func.func @test_greater_than_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                        %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                        %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // CHECK-NOT: d2m.tile_sub
        // CHECK-NOT: d2m.tile_gtz
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.sub_binary_tile_init
        // CHECK: ttkernel.sub_binary_tile
        // CHECK: ttkernel.gtz_tile_init
        // CHECK: ttkernel.gtz_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        %1 = "d2m.tile_gtz"(%0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %1 : !ttype_f32
      }
    }
    return
  }

  func.func @test_greater_equal_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                         %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                         %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // CHECK-NOT: d2m.tile_sub
        // CHECK-NOT: d2m.tile_gez
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.sub_binary_tile_init
        // CHECK: ttkernel.sub_binary_tile
        // CHECK: ttkernel.gez_tile_init
        // CHECK: ttkernel.gez_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        %1 = "d2m.tile_gez"(%0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %1 : !ttype_f32
      }
    }
    return
  }

  func.func @test_less_than_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                     %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                     %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // CHECK-NOT: d2m.tile_sub
        // CHECK-NOT: d2m.tile_ltz
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.sub_binary_tile_init
        // CHECK: ttkernel.sub_binary_tile
        // CHECK: ttkernel.ltz_tile_init
        // CHECK: ttkernel.ltz_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        %1 = "d2m.tile_ltz"(%0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %1 : !ttype_f32
      }
    }
    return
  }

  func.func @test_less_equal_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                      %in1: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>,
                                      %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_f32, #l1_>, memref<1x1x!ttype_f32, #l1_>) outs(%cb2 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32, %arg2: !ttype_f32):
        // CHECK-NOT: d2m.tile_sub
        // CHECK-NOT: d2m.tile_lez
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.sub_binary_tile_init
        // CHECK: ttkernel.sub_binary_tile
        // CHECK: ttkernel.lez_tile_init
        // CHECK: ttkernel.lez_tile
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_f32, !ttype_f32) -> !ttype_f32
        %1 = "d2m.tile_lez"(%0) : (!ttype_f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %1 : !ttype_f32
      }
    }
    return
  }

  //===----------------------------------------------------------------------===//
  // TTIR Comparison operations (i32)
  //===----------------------------------------------------------------------===//

  func.func @test_equal_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                     %in1: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                     %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_si32, #l1_>, memref<1x1x!ttype_si32, #l1_>) outs(%cb2 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32, %arg2: !ttype_si32):
        // CHECK-NOT: d2m.tile_sub
        // CHECK-NOT: d2m.tile_eqz
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.sub_binary_tile_init
        // CHECK: ttkernel.sub_binary_tile
        // CHECK: ttkernel.eqz_tile_init
        // CHECK: ttkernel.eqz_tile_int32
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_si32, !ttype_si32) -> !ttype_si32
        %1 = "d2m.tile_eqz"(%0) : (!ttype_si32) -> !ttype_si32
        // CHECK: ttkernel.pack_tile
        linalg.yield %1 : !ttype_si32
      }
    }
    return
  }

  func.func @test_not_equal_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                         %in1: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                         %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_si32, #l1_>, memref<1x1x!ttype_si32, #l1_>) outs(%cb2 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32, %arg2: !ttype_si32):
        // CHECK-NOT: d2m.tile_sub
        // CHECK-NOT: d2m.tile_nez
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.sub_binary_tile_init
        // CHECK: ttkernel.sub_binary_tile
        // CHECK: ttkernel.nez_tile_init
        // CHECK: ttkernel.nez_tile_int32
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_si32, !ttype_si32) -> !ttype_si32
        %1 = "d2m.tile_nez"(%0) : (!ttype_si32) -> !ttype_si32
        // CHECK: ttkernel.pack_tile
        linalg.yield %1 : !ttype_si32
      }
    }
    return
  }

  func.func @test_greater_than_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                            %in1: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                            %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_si32, #l1_>, memref<1x1x!ttype_si32, #l1_>) outs(%cb2 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32, %arg2: !ttype_si32):
        // CHECK-NOT: d2m.tile_sub
        // CHECK-NOT: d2m.tile_gtz
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.sub_binary_tile_init
        // CHECK: ttkernel.sub_binary_tile
        // CHECK: ttkernel.gtz_tile_init
        // CHECK: ttkernel.gtz_tile_int32
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_si32, !ttype_si32) -> !ttype_si32
        %1 = "d2m.tile_gtz"(%0) : (!ttype_si32) -> !ttype_si32
        // CHECK: ttkernel.pack_tile
        linalg.yield %1 : !ttype_si32
      }
    }
    return
  }

  func.func @test_greater_equal_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                             %in1: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                             %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_si32, #l1_>, memref<1x1x!ttype_si32, #l1_>) outs(%cb2 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32, %arg2: !ttype_si32):
        // CHECK-NOT: d2m.tile_sub
        // CHECK-NOT: d2m.tile_gez
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.sub_binary_tile_init
        // CHECK: ttkernel.sub_binary_tile
        // CHECK: ttkernel.gez_tile_init
        // CHECK: ttkernel.gez_tile_int32
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_si32, !ttype_si32) -> !ttype_si32
        %1 = "d2m.tile_gez"(%0) : (!ttype_si32) -> !ttype_si32
        // CHECK: ttkernel.pack_tile
        linalg.yield %1 : !ttype_si32
      }
    }
    return
  }

  func.func @test_less_than_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                         %in1: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                         %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_si32, #l1_>, memref<1x1x!ttype_si32, #l1_>) outs(%cb2 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32, %arg2: !ttype_si32):
        // CHECK-NOT: d2m.tile_sub
        // CHECK-NOT: d2m.tile_ltz
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.sub_binary_tile_init
        // CHECK: ttkernel.sub_binary_tile
        // CHECK: ttkernel.ltz_tile_init
        // CHECK: ttkernel.ltz_tile_int32
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_si32, !ttype_si32) -> !ttype_si32
        %1 = "d2m.tile_ltz"(%0) : (!ttype_si32) -> !ttype_si32
        // CHECK: ttkernel.pack_tile
        linalg.yield %1 : !ttype_si32
      }
    }
    return
  }

  func.func @test_less_equal_i32_lowering(%in0: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                          %in1: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>,
                                          %out: memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0, %in1 : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>, memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_si32, #ttcore.shard<4096x4096>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>, %arg2_cb: !d2m.cb<memref<1x1x!ttype_si32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb1 = d2m.wait %arg1_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      %cb2 = d2m.reserve %arg2_cb : !d2m.cb<memref<1x1x!ttype_si32, #l1_>> -> memref<1x1x!ttype_si32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0, %cb1 : memref<1x1x!ttype_si32, #l1_>, memref<1x1x!ttype_si32, #l1_>) outs(%cb2 : memref<1x1x!ttype_si32, #l1_>) {
      ^bb0(%arg0: !ttype_si32, %arg1: !ttype_si32, %arg2: !ttype_si32):
        // CHECK-NOT: d2m.tile_sub
        // CHECK-NOT: d2m.tile_lez
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.sub_binary_tile_init
        // CHECK: ttkernel.sub_binary_tile
        // CHECK: ttkernel.lez_tile_init
        // CHECK: ttkernel.lez_tile_int32
        %0 = "d2m.tile_sub"(%arg0, %arg1) : (!ttype_si32, !ttype_si32) -> !ttype_si32
        %1 = "d2m.tile_lez"(%0) : (!ttype_si32) -> !ttype_si32
        // CHECK: ttkernel.pack_tile
        linalg.yield %1 : !ttype_si32
      }
    }
    return
  }
}
