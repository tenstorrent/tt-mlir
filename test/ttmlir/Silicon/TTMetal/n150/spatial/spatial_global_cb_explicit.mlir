// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% ttnn-mode=false" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir
// RUN: ttrt run %t.ttm

// Explicit mapping example: irregular sender (0,0) -> receiver (2,1).

// CHECK-LABEL: func.func @spatial_global_cb_explicit
// CHECK: ttmetal.create_global_circular_buffer
// CHECK-SAME: mapping = #d2m.global_cb_mapping<explicit, [#d2m.sender_receivers<(0,0), [<(2,1), (2,1)>]>]>
// CHECK: "ttmetal.enqueue_program"

#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>
#map1 = affine_map<(d0, d1) -> (0, d0, d1)>
#map2 = affine_map<(d0, d1, d2, d3) -> (d0 + 2, d1 + 1, d2, d3)>
#map3 = affine_map<(d0, d1) -> (0, d0 - 2, d1 - 1)>
#map4 = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
!slot = tensor<1x1x!ttcore.tile<32x32, bf16>>

module {
  func.func @spatial_global_cb_explicit(%arg0: tensor<32x32xbf16>) -> (tensor<32x32xbf16>, tensor<32x32xbf16>) attributes {tt.function_type = "forward_device"} {
    %0 = d2m.empty() {virtualGridForwardMapping = #map, virtualGridInverseMapping = #map1} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %1 = d2m.to_layout %arg0, %0 : tensor<32x32xbf16> into tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %view_in = d2m.view_layout %1 remapping = #map : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %o0 = d2m.empty() {virtualGridForwardMapping = #map, virtualGridInverseMapping = #map1} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %view_out0 = d2m.view_layout %o0 remapping = #map : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %o1 = d2m.empty() {virtualGridForwardMapping = #map2, virtualGridInverseMapping = #map3} : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %view_out1 = d2m.view_layout %o1 remapping = #map : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %gcb = d2m.create_global_cb {
      mapping = #d2m.global_cb_mapping<explicit, [
        #d2m.sender_receivers<(0, 0), [#ttcore.core_range<(2, 1), (2, 1)>]>
      ]>,
      num_slots = 1 : i64
    } : !d2m.global_cb<!slot>
    %r:2 = d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (0, 0)>, #ttcore.core_range<(2, 1), (2, 1)>]}
        ins(%view_in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
        outs(%view_out0, %view_out1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>) {
      ^region0():
        %g0 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map4, #map4], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins(%view_in : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
            outs(%view_out0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
            additionalArgs(%gcb : !d2m.global_cb<!slot>) {
          %block0 = d2m.block_index(0) : index
          %block1 = d2m.block_index(1) : index
          %t0 = tensor.empty() : !slot
          %ld = d2m.remote_load %t0 %view_in[%block0, %block1] : !slot, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> -> !slot
          d2m.global_cb_reserve %gcb : !d2m.global_cb<!slot> -> !slot
          d2m.global_cb_push %gcb, %ld : !d2m.global_cb<!slot>, !slot
          %b0 = d2m.block_index(0) : index
          %b1 = d2m.block_index(1) : index
          %st = d2m.remote_store %view_out0[%b0, %b1] %ld : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>, !slot -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
          d2m.yield %st : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
        d2m.spatial_yield %g0 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
    }, {
      ^region1():
        %g1 = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map4], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
            ins()
            outs(%view_out1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
            additionalArgs(%gcb : !d2m.global_cb<!slot>) {
          %got = d2m.global_cb_wait %gcb : !d2m.global_cb<!slot> -> !slot
          %b0 = d2m.block_index(0) : index
          %b1 = d2m.block_index(1) : index
          %st = d2m.remote_store %view_out1[%b0, %b1] %got : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>, !slot -> tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
          d2m.global_cb_pop %gcb : !d2m.global_cb<!slot>
          d2m.yield %st : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
        } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
        d2m.spatial_yield %g1 : (tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>)
    } : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>, tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout>
    %e0 = d2m.empty() : tensor<32x32xbf16>
    %e1 = d2m.empty() : tensor<32x32xbf16>
    %out0 = d2m.to_layout %r#0, %e0 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> into tensor<32x32xbf16> -> tensor<32x32xbf16>
    %out1 = d2m.to_layout %r#1, %e1 : tensor<1x1x1x1x!ttcore.tile<32x32, bf16>, #layout> into tensor<32x32xbf16> -> tensor<32x32xbf16>
    return %out0, %out1 : tensor<32x32xbf16>, tensor<32x32xbf16>
  }
}
