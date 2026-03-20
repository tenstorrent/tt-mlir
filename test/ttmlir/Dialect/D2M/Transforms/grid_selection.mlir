// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-d2m %s | FileCheck %s --check-prefix=CHECK-BEFORE
// RUN: ttmlir-opt --split-input-file --ttcore-register-device --ttir-to-d2m --d2m-grid-selection --canonicalize %s | FileCheck %s --check-prefix=CHECK-AFTER

#any_device = #ttcore.device<workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1] -> (0, 0, 0, d0 * s1 + d1 * s1 + d2 + s0), meshShape = , chipIds = [0]>

module attributes {ttcore.device = #any_device} {
  func.func @test_grid_selection(%arg0: tensor<256x256xf32>) -> tensor<256x256xf32> {
    // CHECK-BEFORE-LABEL: func.func @test_grid_selection
    // Verify TTIRToD2M creates 1x1 grids
    // CHECK-BEFORE: d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>
    // CHECK-BEFORE: d2m.to_layout %arg0, %{{.*}} : tensor<256x256xf32> into tensor<1x1x8x8x!ttcore.tile<32x32, f32>
    // CHECK-BEFORE: d2m.generic {{{.*}}grid = #ttcore.grid<1x1>

    // CHECK-AFTER-LABEL: func.func @test_grid_selection
    // Verify D2MGridSelection optimizes to 8x8 grids
    // CHECK-AFTER: d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, f32>
    // CHECK-AFTER: d2m.to_layout %arg0, %{{.*}} : tensor<256x256xf32> into tensor<8x8x1x1x!ttcore.tile<32x32, f32>
    // CHECK-AFTER: d2m.generic {{{.*}}grid = #ttcore.grid<8x8>

    %0 = "ttir.exp"(%arg0) : (tensor<256x256xf32>) -> tensor<256x256xf32>
    return %0 : tensor<256x256xf32>
  }
}

// -----

#layout = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, undef, l1, sharded>

module {
  func.func @test_update_empty() -> (tensor<256x256xf32>) {
    // CHECK-BEFORE-LABEL: func.func @test_update_empty
    // CHECK-BEFORE: d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>
    // CHECK-AFTER-LABEL: func.func @test_update_empty
    // Verify D2MGridSelection optimizes the empty 8x8 grids
    // CHECK-AFTER: d2m.empty() : tensor<8x8x1x1x!ttcore.tile<32x32, f32>
    %0 = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>

    %1 = d2m.generic {
      block_factors = [1, 1],
      grid = #ttcore.grid<1x1>,
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>],
      iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
      threads = [#d2m.thread<unified>]
    }
    ins() outs(%0 : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>)  {
    ^unified0:
      %cb_out = d2m.get_cb(0) : !d2m.cb<tensor<8x8x!ttcore.tile<32x32, f32>>>
      %out = tensor.empty() : tensor<8x8x!ttcore.tile<32x32, f32>>
      d2m.yield %out : (tensor<8x8x!ttcore.tile<32x32, f32>>)
    } : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout>

    %2 = d2m.empty() : tensor<256x256xf32>
    %3 = d2m.to_layout %1, %2 : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout> into tensor<256x256xf32> -> tensor<256x256xf32>

    return %3 : tensor<256x256xf32>
  }
}

 // -----

 #layout_tm_device_input = #ttcore.metal_layout<logical_shape = 33x2x8, dim_alignments = 1x32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>
 #layout_tm_stream_plain = #ttcore.metal_layout<logical_shape = 2x264, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>
// CHECK-AFTER: #layout1 = #ttcore.metal_layout<logical_shape = 2x264, dim_alignments = 32x256, collapsed_intervals = dense<> : tensor<0x2xi64>, {{.*}}>
 #layout_tm_stream_map = #ttcore.metal_layout<logical_shape = 2x264, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>

 module {
   // Test to make sure aligned tensor shapes is used in TM affine maps when we start with unaligned stream output.
   func.func @test_unaligned_tm(%arg0: tensor<33x2x8xf32>) -> tensor<2x264xf32> {
     // CHECK-AFTER-LABEL: func.func @test_unaligned_tm
     // CHECK-AFTER-NOT: 288
     %device_storage = d2m.empty() : tensor<1x1x1x33x32x32xf32, #layout_tm_device_input>
     %device_input = d2m.to_layout %arg0, %device_storage : tensor<33x2x8xf32> into tensor<1x1x1x33x32x32xf32, #layout_tm_device_input> -> tensor<1x1x1x33x32x32xf32, #layout_tm_device_input>
     %stream_storage = d2m.empty() : tensor<1x1x32x288xf32, #layout_tm_stream_plain>
    // CHECK-AFTER: "d2m.stream_layout"{{.*}} -> tensor<1x8x32x64xf32, #layout1>
     %stream = "d2m.stream_layout"(%device_input, %stream_storage) <{remapping = affine_map<(d0, d1, d2, d3) -> (0, d0, d1, d3 floordiv 8, d2, d3 mod 8)>}> : (tensor<1x1x1x33x32x32xf32, #layout_tm_device_input>, tensor<1x1x32x288xf32, #layout_tm_stream_plain>) -> tensor<1x1x32x288xf32, #layout_tm_stream_map>
     %host_output = d2m.empty() : tensor<2x264xf32>
     %stream_output = d2m.empty() : tensor<1x1x32x288xf32, #layout_tm_stream_plain>
     %device_output = d2m.generic {
       block_factors = [1, 1],
       grid = #ttcore.grid<1x1>,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
       threads = [#d2m.thread<unified>]}
         ins(%stream : tensor<1x1x32x288xf32, #layout_tm_stream_map>)
         outs(%stream_output : tensor<1x1x32x288xf32, #layout_tm_stream_plain>)  {
     ^unified:
       %cb0 = d2m.get_cb(0) : !d2m.cb<tensor<32x288xf32>>
       %cb1 = d2m.get_cb(1) : !d2m.cb<tensor<32x288xf32>>
       %i = d2m.block_index(0) : index
       %j = d2m.block_index(1) : index
       %buffer = tensor.empty() : tensor<32x288xf32>
       %load_result = d2m.remote_load %buffer %stream[%i, %j]: tensor<32x288xf32>, tensor<1x1x32x288xf32, #layout_tm_stream_map> -> tensor<32x288xf32>
       d2m.yield %load_result : (tensor<32x288xf32>)
     } : tensor<1x1x32x288xf32, #layout_tm_stream_plain>
     %output = d2m.to_layout %device_output, %host_output : tensor<1x1x32x288xf32, #layout_tm_stream_plain> into tensor<2x264xf32> -> tensor<2x264xf32>
     return %output : tensor<2x264xf32>
   }
 }

 // -----

// CHECK-AFTER: #layout1 = #ttcore.metal_layout<{{.*}}>
 #layout_in = #ttcore.metal_layout<logical_shape = 64x64, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>
 #layout_out = #ttcore.metal_layout<logical_shape = 40x40, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>
 #layout_op = #ttcore.metal_layout<logical_shape = 40x40, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>

 module {
   // Test to make sure the reblock map contains no unnecessary terms.
   func.func @test_simplified_reblock_map(%arg0: tensor<64x64xbf16>) -> tensor<40x40xbf16> {
     // CHECK-AFTER-LABEL: func.func @test_simplified_reblock_map
     %device_storage = d2m.empty() : tensor<1x1x64x64xbf16, #layout_in>
    // CHECK-AFTER: %[[STREAM:.*]] = "d2m.stream_layout"{{.*}} -> tensor<2x2x32x32xbf16, #layout1>
     %device_input = d2m.to_layout %arg0, %device_storage : tensor<64x64xbf16> into tensor<1x1x64x64xbf16, #layout_in> -> tensor<1x1x64x64xbf16, #layout_in>
     %stream_storage = d2m.empty() : tensor<1x1x64x64xbf16, #layout_out>
    %stream = "d2m.stream_layout"(%device_input, %stream_storage) <{remapping = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2 + 10, d3 + 20)>}> : (tensor<1x1x64x64xbf16, #layout_in>, tensor<1x1x64x64xbf16, #layout_out>) -> tensor<1x1x64x64xbf16, #layout_op>
     %host_output = d2m.empty() : tensor<40x40xbf16>
     %stream_output = d2m.empty() : tensor<1x1x64x64xbf16, #layout_out>
     %device_output = d2m.generic {
       block_factors = [1, 1],
       grid = #ttcore.grid<1x1>,
       indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>],
       iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>],
       threads = [#d2m.thread<unified>]}
         ins(%stream : tensor<1x1x64x64xbf16, #layout_op>)
         outs(%stream_output : tensor<1x1x64x64xbf16, #layout_out>)  {
     ^unified:
       %cb0 = d2m.get_cb(0) : !d2m.cb<tensor<64x64xbf16>>
       %cb1 = d2m.get_cb(1) : !d2m.cb<tensor<64x64xbf16>>
       %i = d2m.block_index(0) : index
       %j = d2m.block_index(1) : index
       %buffer = tensor.empty() : tensor<64x64xbf16>
       %r = d2m.remote_load %buffer %stream[%i, %j] : tensor<64x64xbf16>, tensor<1x1x64x64xbf16, #layout_op> -> tensor<64x64xbf16>
       d2m.yield %r : (tensor<64x64xbf16>)
     } : tensor<1x1x64x64xbf16, #layout_out>
     %output = d2m.to_layout %device_output, %host_output : tensor<1x1x64x64xbf16, #layout_out> into tensor<40x40xbf16> -> tensor<40x40xbf16>
     return %output : tensor<40x40xbf16>
   }
 }

// -----

#layout_in_S = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>
#layout_out_S = #ttcore.metal_layout<logical_shape = 32x64, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>

module  {
  func.func @test_reblock_composite_view_small(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x64xf32> {
    // CHECK-AFTER-LABEL: func.func @test_reblock_composite_view_small
    %0 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_in_S>
    %1 = d2m.to_layout %arg0, %0 : tensor<32x32xf32> into tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_in_S> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_in_S>
    %2 = d2m.empty() : tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_in_S>
    %3 = d2m.to_layout %arg1, %2 : tensor<32x32xf32> into tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_in_S> -> tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_in_S>
    // CHECK-AFTER: "d2m.composite_view"{{.+}} -> tensor<1x2x1x1x!ttcore.tile<32x32, f32>
    %4 = "d2m.composite_view"(%1, %3) <{dim = 1 : si32}> : (tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_in_S>, tensor<1x1x1x1x!ttcore.tile<32x32, f32>, #layout_in_S>) -> tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout_out_S>
    %5 = d2m.empty() : tensor<32x64xf32>
    %6 = d2m.empty() : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout_out_S>
    // CHECK-AFTER: d2m.generic {{.+}} grid = #ttcore.grid<1x2>
    %7 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
        ins(%4 : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout_out_S>)
        outs(%6 : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout_out_S>)
     {
      %9 = tensor.empty() : tensor<1x2x!ttcore.tile<32x32, f32>>
      %10 = tensor.empty() : tensor<1x2x!ttcore.tile<32x32, f32>>
      %block0 = d2m.block_index(0) : index
      %block1 = d2m.block_index(1) : index
      %11 = d2m.remote_load %9 %4[%block0, %block1] : tensor<1x2x!ttcore.tile<32x32, f32>>, tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout_out_S> -> tensor<1x2x!ttcore.tile<32x32, f32>>
      %12 = d2m.remote_store %6[%block0, %block1] %11 : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout_out_S>, tensor<1x2x!ttcore.tile<32x32, f32>> -> tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout_out_S>
      d2m.yield %12 : (tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout_out_S>)
    } : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout_out_S>
    %8 = d2m.to_layout %7, %5 : tensor<1x1x1x2x!ttcore.tile<32x32, f32>, #layout_out_S> into tensor<32x64xf32> -> tensor<32x64xf32>
    return %8 : tensor<32x64xf32>
  }
}

// -----

#layout_in_L = #ttcore.metal_layout<logical_shape = 256x256, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>
#layout_out_L = #ttcore.metal_layout<logical_shape = 256x512, dim_alignments = 32x32, collapsed_intervals = dense<> : tensor<0x2xi64>, undef, l1, sharded>

module  {
  func.func @test_reblock_composite_view_large(%arg0: tensor<256x256xf32>, %arg1: tensor<256x256xf32>) -> tensor<256x512xf32> attributes {tt.function_type = "forward_device"} {
    // CHECK-AFTER-LABEL: func.func @test_reblock_composite_view_large
    %0 = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout_in_L>
    %1 = d2m.to_layout %arg0, %0 : tensor<256x256xf32> into tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout_in_L> -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout_in_L>
    %2 = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout_in_L>
    %3 = d2m.to_layout %arg1, %2 : tensor<256x256xf32> into tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout_in_L> -> tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout_in_L>
    // CHECK-AFTER: "d2m.composite_view"{{.+}} -> tensor<8x8x1x2x!ttcore.tile<32x32, f32>
    %4 = "d2m.composite_view"(%1, %3) <{dim = 1 : si32}> : (tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout_in_L>, tensor<1x1x8x8x!ttcore.tile<32x32, f32>, #layout_in_L>) -> tensor<1x1x8x16x!ttcore.tile<32x32, f32>, #layout_out_L>
    %5 = d2m.empty() : tensor<256x512xf32>
    %6 = d2m.empty() : tensor<1x1x8x16x!ttcore.tile<32x32, f32>, #layout_out_L>
    // CHECK-AFTER: d2m.generic {{.+}} grid = #ttcore.grid<8x8>
    %7 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#ttcore.iterator_type<parallel>, #ttcore.iterator_type<parallel>], threads = [#d2m.thread<unified>]}
        ins(%4 : tensor<1x1x8x16x!ttcore.tile<32x32, f32>, #layout_out_L>)
        outs(%6 : tensor<1x1x8x16x!ttcore.tile<32x32, f32>, #layout_out_L>)
     {
      %9 = tensor.empty() : tensor<8x16x!ttcore.tile<32x32, f32>>
      %10 = tensor.empty() : tensor<8x16x!ttcore.tile<32x32, f32>>
      %block0 = d2m.block_index(0) : index
      %block1 = d2m.block_index(1) : index
      %11 = d2m.remote_load %9 %4[%block0, %block1] : tensor<8x16x!ttcore.tile<32x32, f32>>, tensor<1x1x8x16x!ttcore.tile<32x32, f32>, #layout_out_L> -> tensor<8x16x!ttcore.tile<32x32, f32>>
      %12 = d2m.remote_store %6[%block0, %block1] %11 : tensor<1x1x8x16x!ttcore.tile<32x32, f32>, #layout_out_L>, tensor<8x16x!ttcore.tile<32x32, f32>> -> tensor<1x1x8x16x!ttcore.tile<32x32, f32>, #layout_out_L>
      d2m.yield %12 : (tensor<1x1x8x16x!ttcore.tile<32x32, f32>, #layout_out_L>)
    } : tensor<1x1x8x16x!ttcore.tile<32x32, f32>, #layout_out_L>
    %8 = d2m.to_layout %7, %5 : tensor<1x1x8x16x!ttcore.tile<32x32, f32>, #layout_out_L> into tensor<256x512xf32> -> tensor<256x512xf32>
    return %8 : tensor<256x512xf32>
  }
}
