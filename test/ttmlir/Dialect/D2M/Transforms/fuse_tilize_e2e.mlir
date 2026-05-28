// End-to-end check: a D2M func with one row-major and one already-tiled
// operand flows through the canonical `--ttir-to-ttmetal-pipeline` and lowers
// to the fused on-the-fly tilize + per-tile matmul form.
//
// Why D2M-level (not bare TTIR): the row-major vs tiled distinction has to be
// expressed in the func signature's encoding. Bare `ttir.matmul` operands are
// untyped scalar tensors, and TTIRToD2M then picks all-tiled by default, so
// no pure row-major→tile `d2m.to_layout` survives for the fuse pass to match.
//
// `override-device-shape=1,1` makes GridSelection a no-op so the input's
// `1x1x...` sharded form survives. `use-tile-matmul=true` keeps the matmul on
// the per-tile (`mm_init` + `matmul_tiles`) path.
//
// Tests the dt-restore wiring: when the fuse pass folds the tilize into the
// matmul generic, the same compute kernel runs `tilize_block` and then
// `matmul_tiles`. Because the tilize writes into the matmul's srcA CB,
// `mm_init_short_with_dt` must flip the unpacker srcA data format back to the
// tiled scratch each K iteration.

// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% override-device-shape=1,1 use-tile-matmul=true" -o %t %s
// RUN: FileCheck %s --input-file=%t

#layout    = #ttcore.metal_layout<logical_shape = 128x96, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layout1   = #ttcore.metal_layout<logical_shape = 96x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layout2   = #ttcore.metal_layout<logical_shape = 128x64, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>

#map  = affine_map<(d0, d1, d2) -> (d0, d2)>
#map1 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map2 = affine_map<(d0, d1, d2) -> (d0, d1)>

#parallel  = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

// CHECK: func.func @rm_fed_matmul

// DM kernel issues an NoC read of the row-major source into the f32 CB.
// CHECK-DAG: noc.async_read

// Compute kernel tilizes the row-major CB into the tile scratch CB on the
// fly, then runs per-tile matmul with `mm_init_short_with_dt` restoring the
// unpacker srcA data format each K iteration. `experimental::matmul_block`
// must NOT appear.
// CHECK-DAG: call_opaque "tilize_init"
// CHECK-DAG: call_opaque "experimental::tilize_block"
// CHECK-DAG: call_opaque "mm_init"
// CHECK-DAG: call_opaque "mm_init_short_with_dt"
// CHECK-DAG: call_opaque "matmul_tiles"
// CHECK-NOT: experimental::matmul_block

func.func @rm_fed_matmul(%rm_a: tensor<1x1x128x96xf32, #layout>, %tile_b: tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout1>) -> tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout2> {
  %tiled_empty = d2m.empty() : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>
  %tiled_a = d2m.to_layout %rm_a, %tiled_empty : tensor<1x1x128x96xf32, #layout> into tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout> -> tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>
  %out_empty = d2m.empty() : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout2>
  %r = d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map1, #map2], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
      ins(%tiled_a, %tile_b : tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout>, tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout1>)
      outs(%out_empty : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout2>) {
    %b0 = d2m.block_index(0) : index
    %b1 = d2m.block_index(1) : index
    %b2 = d2m.block_index(2) : index
    %buf_a = tensor.empty() : tensor<4x3x!ttcore.tile<32x32, f32>>
    %la = d2m.remote_load %buf_a %tiled_a[%b0, %b2] : tensor<4x3x!ttcore.tile<32x32, f32>>, tensor<1x1x4x3x!ttcore.tile<32x32, f32>, #layout> -> tensor<4x3x!ttcore.tile<32x32, f32>>
    %buf_b = tensor.empty() : tensor<3x2x!ttcore.tile<32x32, f32>>
    %lb = d2m.remote_load %buf_b %tile_b[%b2, %b1] : tensor<3x2x!ttcore.tile<32x32, f32>>, tensor<1x1x3x2x!ttcore.tile<32x32, f32>, #layout1> -> tensor<3x2x!ttcore.tile<32x32, f32>>
    %buf_out = tensor.empty() : tensor<4x2x!ttcore.tile<32x32, f32>>
    %mm = linalg.generic {indexing_maps = [#map, #map1, #map2], iterator_types = ["parallel", "parallel", "reduction"]} ins(%la, %lb : tensor<4x3x!ttcore.tile<32x32, f32>>, tensor<3x2x!ttcore.tile<32x32, f32>>) outs(%buf_out : tensor<4x2x!ttcore.tile<32x32, f32>>) {
    ^bb0(%in: !ttcore.tile<32x32, f32>, %in_b: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
      %p = "d2m.tile_matmul"(%in, %in_b, %out) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
      linalg.yield %p : !ttcore.tile<32x32, f32>
    } -> tensor<4x2x!ttcore.tile<32x32, f32>>
    %bo0 = d2m.block_index(0) : index
    %bo1 = d2m.block_index(1) : index
    %s = d2m.remote_store %out_empty[%bo0, %bo1] %mm : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout2>, tensor<4x2x!ttcore.tile<32x32, f32>> -> tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout2>
    d2m.yield %s : (tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout2>)
  } : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout2>
  return %r : tensor<1x1x4x2x!ttcore.tile<32x32, f32>, #layout2>
}
