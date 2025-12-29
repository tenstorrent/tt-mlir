// UNSUPPORTED: true
// RUN: ttmlir-opt --ttcore-register-device --ttir-to-ttmetal-me-pipeline --convert-d2m-to-ttkernel --canonicalize -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

#l1_ = #ttcore.memory_space<l1>
#map_ = affine_map<(d0, d1) -> (d0, d1)>
#parallel_ = #ttcore.iterator_type<parallel>

!ttype_f32 = !ttcore.tile<32x32, f32>

module {
  //===----------------------------------------------------------------------===//
  // Binary tile ops with f32 scalar RHS operands
  //===----------------------------------------------------------------------===//

  func.func @test_add_scalar_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                      %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        %scalar = arith.constant 2.5 : f32
        // CHECK-NOT: d2m.tile_add
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.binop_with_scalar_tile_init
        // CHECK: ttkernel.add_unary_tile
        %0 = "d2m.tile_add"(%arg0, %scalar) : (!ttype_f32, f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_sub_scalar_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                      %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        %scalar = arith.constant 1.5 : f32
        // CHECK-NOT: d2m.tile_sub
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.binop_with_scalar_tile_init
        // CHECK: ttkernel.sub_unary_tile
        %0 = "d2m.tile_sub"(%arg0, %scalar) : (!ttype_f32, f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_mul_scalar_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                      %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        %scalar = arith.constant 3.0 : f32
        // CHECK-NOT: d2m.tile_mul
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.binop_with_scalar_tile_init
        // CHECK: ttkernel.mul_unary_tile
        %0 = "d2m.tile_mul"(%arg0, %scalar) : (!ttype_f32, f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_div_scalar_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                      %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        %scalar = arith.constant 2.0 : f32
        // CHECK-NOT: d2m.tile_div
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.binop_with_scalar_tile_init
        // CHECK: ttkernel.div_unary_tile
        %0 = "d2m.tile_div"(%arg0, %scalar) : (!ttype_f32, f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

  func.func @test_pow_scalar_lowering(%in0: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>,
                                      %out: memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>) {
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map_, #map_], iterator_types = [#parallel_, #parallel_], threads = [#d2m.thread<compute>]}
        ins(%in0 : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)
        outs(%out : memref<1x1x1x1x!ttype_f32, #ttcore.shard<4096x4096, 1>, #l1_>)  {
    ^compute0(%arg0_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>, %arg1_cb: !d2m.cb<memref<1x1x!ttype_f32, #l1_>>):
      %cb0 = d2m.wait %arg0_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      %cb1 = d2m.reserve %arg1_cb : !d2m.cb<memref<1x1x!ttype_f32, #l1_>> -> memref<1x1x!ttype_f32, #l1_>
      linalg.generic {indexing_maps = [#map_, #map_], iterator_types = ["parallel", "parallel"]} ins(%cb0 : memref<1x1x!ttype_f32, #l1_>) outs(%cb1 : memref<1x1x!ttype_f32, #l1_>) {
      ^bb0(%arg0: !ttype_f32, %arg1: !ttype_f32):
        %scalar = arith.constant 2.0 : f32
        // CHECK-NOT: d2m.tile_pow
        // CHECK: ttkernel.init_sfpu
        // CHECK: ttkernel.power_tile_init
        // CHECK: ttkernel.power_tile
        %0 = "d2m.tile_pow"(%arg0, %scalar) : (!ttype_f32, f32) -> !ttype_f32
        // CHECK: ttkernel.pack_tile
        linalg.yield %0 : !ttype_f32
      }
    }
    return
  }

}
