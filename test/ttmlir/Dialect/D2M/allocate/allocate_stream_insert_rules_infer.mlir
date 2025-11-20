// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=stream-insert-policy=infer" -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify several supported use cases for operand stream insertion by d2m-allocate in stream-insert-policy=infer mode (current default).

#l1_ttnn = #ttnn.buffer_type<l1>
#dram_ttnn = #ttnn.buffer_type<dram>

#l1 = #ttcore.memory_space<l1>
#dram = #ttcore.memory_space<dram>

#map = affine_map<(d0, d1) -> (d0, d1)>
#mapL = affine_map<(d0, d1, d2) -> (d0, d2)>
#mapR = affine_map<(d0, d1, d2) -> (d2, d1)>
#mapO = affine_map<(d0, d1, d2) -> (d0, d1)>

#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

#l1_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1, (d0, d1) -> (0, d0, d1)>, memref<1x1x!ttcore.tile<32x32, f32>, #l1_ttnn>, <block_sharded>>
#dram_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram_ttnn>, <interleaved>>

module {

  // Verify operand stream insertion for all generic inputs.
  // CHECK-LABEL: func @test_generic_insert_missing_streams
  func.func @test_generic_insert_missing_streams() {
    %lhs = memref.alloc() : memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %r = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    // expect two streams inserted by the pass, for %lhs and %rhs, in operand order:
    // CHECK: %[[STREAM_LHS:.*]] = "d2m.stream_layout"({{.+}}) : (memref<1x1x2x3
    // CHECK: %[[STREAM_RHS:.*]] = "d2m.stream_layout"({{.+}}) : (memref<1x1x3x4
    // CHECK: ins(%[[STREAM_LHS]], %[[STREAM_RHS]] :
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%lhs, %rhs : memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #l1>, memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%r : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^compute0(%cb0: !d2m.cb<memref<2x3x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<3x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %0 = d2m.wait %cb0 : <memref<2x3x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x3x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.wait %cb1 : <memref<3x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<3x4x!ttcore.tile<32x32, f32>, #l1>
      %2 = d2m.reserve %cb2 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      "d2m.tile_matmul_block"(%0, %1, %2) : (memref<2x3x!ttcore.tile<32x32, f32>, #l1>, memref<3x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> ()
    }
    return
  }

  // Verify that it is ok for incoming IR to have some operand streams already inserted.
  // CHECK-LABEL: func @test_generic_accept_operand_streams
  func.func @test_generic_accept_operand_streams() {
    %lhs = memref.alloc() : memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %r = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    // expect two streams, for lhs and rhs, in operand order; one of them should be %stream_lhs, the other inserted by the pass:
    // CHECK: %[[STREAM_LHS:.*]] = "d2m.stream_layout"({{.+}}) : (memref<1x1x2x3
    // CHECK: %[[STREAM_RHS:.*]] = "d2m.stream_layout"({{.+}}) : (memref<1x1x3x4
    // CHECK: ins(%[[STREAM_LHS]], %[[STREAM_RHS]] :
    %buf_lhs = memref.alloc() : memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 2>, #l1>
    %stream_lhs = "d2m.stream_layout"(%lhs, %buf_lhs) : (memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #l1>, memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 2>, #l1>) -> memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<compute>]}
        ins(%stream_lhs, %rhs : memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #l1>, memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%r : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^compute0(%cb0: !d2m.cb<memref<2x3x!ttcore.tile<32x32, f32>, #l1>>, %cb1: !d2m.cb<memref<3x4x!ttcore.tile<32x32, f32>, #l1>>, %cb2: !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>):
      %0 = d2m.wait %cb0 : <memref<2x3x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x3x!ttcore.tile<32x32, f32>, #l1>
      %1 = d2m.wait %cb1 : <memref<3x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<3x4x!ttcore.tile<32x32, f32>, #l1>
      %2 = d2m.reserve %cb2 : <memref<2x4x!ttcore.tile<32x32, f32>, #l1>> -> memref<2x4x!ttcore.tile<32x32, f32>, #l1>
      "d2m.tile_matmul_block"(%0, %1, %2) : (memref<2x3x!ttcore.tile<32x32, f32>, #l1>, memref<3x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> ()
    }
    return
  }

  // Verify that "DMA-only" generics do not have operand streams inserted.
  // CHECK-LABEL: func @test_generic_dma_only
  func.func @test_generic_dma_only(%arg0: memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<dram>>) {
    // CHECK-NOT: d2m.stream_layout
    %view_arg0 = d2m.view_layout %arg0 : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<dram>> -> memref<1x1x32x32xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>
    %out = memref.alloc() : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<datamovement>]}
        ins(%view_arg0 : memref<1x1x32x32xf32, #ttcore.view<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>, #ttcore.memory_space<dram>>)
        outs(%out : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>)  {
    ^datamovement0(%cb0: !d2m.cb<memref<32x32xf32, #ttcore.memory_space<dram>>>, %cb1: !d2m.cb<memref<32x32xf32, #ttcore.memory_space<l1>>>):
      %buf = d2m.reserve %cb1 : !d2m.cb<memref<32x32xf32, #ttcore.memory_space<l1>>> -> memref<32x32xf32, #ttcore.memory_space<l1>>
    }
    return
  }

  // Verify that "explicit datamovement" generics must insert their own operand streams.
  // CHECK-LABEL: func @test_generic_explicit_datamovement
  // CHECK: (%[[ARG_IN:.*]]: memref<{{.+}}>, %[[ARG_OUT:.*]]: memref<{{.+}}>)
  func.func @test_generic_explicit_datamovement(%arg_in: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>, %arg_out: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    // CHECK: %[[ALLOC_IN:.*]] = memref.alloc
    // CHECK: %[[STREAM_IN:.*]] = "d2m.stream_layout"(%[[ARG_IN]], %[[ALLOC_IN]]
    %buf_in = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    %stream_in = "d2m.stream_layout"(%arg_in, %buf_in) : (memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>
    // expect %stream_in stream but no stream for %arg_out (no output streams by default):
    // CHECK: ins(%[[STREAM_IN]] :
    // CHECK: outs(%[[ARG_OUT]] :
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<compute>]}
        ins(%stream_in : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<map(4)>, #dram>)
        outs(%arg_out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>) {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #dram>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>):
      %val = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #dram>> -> memref<1x1x!ttcore.tile<32x32, f32>, #dram>
      %out = d2m.reserve %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }

  // Verify that the pass works with ttir.ttnn_metal_layout_casts.
  // CHECK-LABEL: func @test_ttnn_arg_cast_bridge
  func.func @test_ttnn_arg_cast_bridge(%arg0: tensor<32x32xf32, #dram_layout>, %arg1: tensor<32x32xf32, #l1_layout>)  {
    // CHECK-NOT: d2m.stream_layout
    %arg0_cast = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x32xf32, #dram_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
    %arg1_cast = ttir.ttnn_metal_layout_cast %arg1 : tensor<32x32xf32, #l1_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<compute>]}
        ins(%arg0_cast : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>)
        outs(%arg1_cast : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)  {
    ^compute0(%cb0: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #dram>>, %cb1: !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>):
      %in = d2m.wait %cb0 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #dram>> -> memref<1x1x!ttcore.tile<32x32, f32>, #dram>
      %out = d2m.reserve %cb1 : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
    }
    return
  }

} // module
