// RUN: ttmlir-opt --ttcore-register-device "--d2m-allocate=stream-insert-policy=infer test-buffer-size-policy=max" -o %t %s
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
#remap4 = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#remap_dma = affine_map<(d0, d1, d2, d3) -> (d0 + d2 floordiv 32, d1 + d3 floordiv 32, d2 mod 32, d3 mod 32)>

#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

#l1_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1_ttnn>, <block_sharded>, core_ranges = <[#ttnn.core_range<(0, 0), (0, 0)>]>>
#dram_layout = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #dram_ttnn>, <interleaved>>

module {

  // Verify operand stream insertion for all generic inputs.
  // CHECK-LABEL: func @test_generic_insert_missing_streams
    // CHECK: cb_layout
  func.func @test_generic_insert_missing_streams() {
    %lhs = memref.alloc() : memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #l1>
    %rhs = memref.alloc() : memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    %r = memref.alloc() : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
    // expect two streams inserted by the pass, for %lhs and %rhs, in operand order:
    d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#mapL, #mapR, #mapO], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
        ins(%lhs, %rhs : memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #l1>, memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)
        outs(%r : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>)  {
    ^unified0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<2x3x!ttcore.tile<32x32, f32>, #l1>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<3x4x!ttcore.tile<32x32, f32>, #l1>>
      %cb2 = d2m.get_cb(2) : !d2m.cb<memref<2x4x!ttcore.tile<32x32, f32>, #l1>>
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      %bf2 = d2m.get_block_factor(2) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          affine.for %iter2 = 0 to %bf2 {
            %buffer_lhs = memref.alloc() : memref<2x3x!ttcore.tile<32x32, f32>, #l1>
            %0 = d2m.remote_load %buffer_lhs %lhs[%iter0, %iter2] : memref<2x3x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x2x3x!ttcore.tile<32x32, f32>, #ttcore.shard<12288x4096, 1>, #l1> -> memref<2x3x!ttcore.tile<32x32, f32>, #l1>
            %buffer_rhs = memref.alloc() : memref<3x4x!ttcore.tile<32x32, f32>, #l1>
            %1 = d2m.remote_load %buffer_rhs %rhs[%iter2, %iter1] : memref<3x4x!ttcore.tile<32x32, f32>, #l1>, memref<1x1x3x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1> -> memref<3x4x!ttcore.tile<32x32, f32>, #l1>
            %buffer_out = memref.alloc() : memref<2x4x!ttcore.tile<32x32, f32>, #l1>
            "d2m.tile_matmul_block"(%0, %1, %buffer_out) : (memref<2x3x!ttcore.tile<32x32, f32>, #l1>, memref<3x4x!ttcore.tile<32x32, f32>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1>) -> ()
            %result = d2m.remote_store %r[%iter0, %iter1] %buffer_out : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>, memref<2x4x!ttcore.tile<32x32, f32>, #l1> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.shard<16384x4096, 1>, #l1>
          } {d2m.blocking_loop = 2}
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }


  // Verify that a local buffer shared by a remote_load and remote_store is
  // treated as aliased (no CB alloc) buffer if one of the operands (input or output) is aliased.
  // CHECK-LABEL: func @test_shared_load_store_buffer_no_cb_if_one_operand_aliased
  // CHECK-NOT: %[[BUFFER:.*]] = memref.alloc() {{.*}} : memref<32x32xf32, #ttcore.cb_layout<128x4, 2>, #l1>
  // CHECK-NOT: %[[LOAD:.*]] = d2m.remote_load {{.*}} : memref<32x32xf32, #ttcore.cb_layout<128x4, 2>, #l1>
  // CHECK-NOT: d2m.remote_store {{.*}} : memref<1x1x32x32xf32, {{.*}} #ttcore.cb_layout<128x4, 2>, #l1>
  func.func @test_shared_load_store_buffer_no_cb_if_one_operand_aliased(%arg0: memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<dram>>) {
    %view_arg0 = d2m.view_layout %arg0 remapping = #remap_dma : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<dram>> -> memref<1x1x32x32xf32, #ttcore.view<4>, #ttcore.memory_space<dram>>
    %out = memref.alloc() : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
        ins(%view_arg0 : memref<1x1x32x32xf32, #ttcore.view<4>, #ttcore.memory_space<dram>>)
        outs(%out : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>)  {
    ^unified0:
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          %buffer = memref.alloc() : memref<32x32xf32, #ttcore.memory_space<l1>>
          %loaded = d2m.remote_load %buffer %view_arg0[%iter0, %iter1] : memref<32x32xf32, #ttcore.memory_space<l1>>, memref<1x1x32x32xf32, #ttcore.view<4>, #ttcore.memory_space<dram>> -> memref<32x32xf32, #ttcore.memory_space<l1>>
          %stored = d2m.remote_store %out[%iter0, %iter1] %buffer : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #ttcore.memory_space<l1>>, memref<32x32xf32, #ttcore.memory_space<l1>> -> memref<32x32xf32, #ttcore.memory_space<l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Verify that a local buffer shared by a remote_load and remote_store is
  // treated as CB-backed if both operands are streamed.
  // CHECK-LABEL: func @test_shared_load_store_buffer_has_cb_if_both_operands_are_streamed
  // CHECK: %[[BUFFER:.*]] = memref.alloc(){{.*}} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
  // CHECK: %[[LOAD:.*]] = d2m.remote_load %[[BUFFER]] {{.*}} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #dram> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1>
  // CHECK: d2m.remote_store {{.*}} %[[BUFFER]] : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #l1> -> memref<1x1x!ttcore.tile<32x32, f32>, #l1>
  func.func @test_shared_load_store_buffer_has_cb_if_both_operands_are_streamed(%arg0: memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<dram>>) {
    %view_arg0 = d2m.view_layout %arg0 remapping = #remap4 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<dram>> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<dram>>
    %out = memref.alloc() : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<dram>>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
        ins(%view_arg0 : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<dram>>)
        outs(%out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<dram>>)  {
    ^unified0:
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          %buffer = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
          %loaded = d2m.remote_load %buffer %view_arg0[%iter0, %iter1] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<dram>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
          %c0 = arith.constant 0 : index
          %elt = memref.load %buffer[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
          %abs = "d2m.tile_abs"(%elt) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
          memref.store %abs, %buffer[%c0, %c0] : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
          %stored = d2m.remote_store %out[%iter0, %iter1] %buffer : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #ttcore.memory_space<dram>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>> -> memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.memory_space<l1>>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

  // Verify that the pass works with ttir.ttnn_metal_layout_casts.
  // CHECK-LABEL: func @test_ttnn_arg_cast_bridge
  func.func @test_ttnn_arg_cast_bridge(%arg0: tensor<32x32xf32, #dram_layout>, %arg1: tensor<32x32xf32, #l1_layout>)  {
    %arg0_cast = ttir.ttnn_metal_layout_cast %arg0 : tensor<32x32xf32, #dram_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>
    %arg1_cast = ttir.ttnn_metal_layout_cast %arg1 : tensor<32x32xf32, #l1_layout> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
    d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%arg0_cast : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram>)
        outs(%arg1_cast : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>)  {
    ^unified0:
      %cb0 = d2m.get_cb(0) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #dram>>
      %cb1 = d2m.get_cb(1) : !d2m.cb<memref<1x1x!ttcore.tile<32x32, f32>, #l1>>
      %bf0 = d2m.get_block_factor(0) : index
      %bf1 = d2m.get_block_factor(1) : index
      affine.for %iter0 = 0 to %bf0 {
        affine.for %iter1 = 0 to %bf1 {
          %buffer_in = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #dram>
          %in = d2m.remote_load %buffer_in %arg0_cast[%iter0, %iter1] : memref<1x1x!ttcore.tile<32x32, f32>, #dram>, memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #dram> -> memref<1x1x!ttcore.tile<32x32, f32>, #dram>
          %buffer_out = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>, #l1>
          %result = d2m.remote_store %arg1_cast[%iter0, %iter1] %buffer_out : memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>, memref<1x1x!ttcore.tile<32x32, f32>, #l1> -> memref<1x1x1x1x!ttcore.tile<32x32, f32>, #ttcore.shard<4096x4096, 1>, #l1>
        } {d2m.blocking_loop = 1}
      } {d2m.blocking_loop = 0}
    }
    return
  }

} // module
