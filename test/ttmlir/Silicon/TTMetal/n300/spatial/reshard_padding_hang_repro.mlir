// Reproducer for the K-padding reshard infinite hang on ring-fabric hardware.
//
// This file is the n300 (2-device ring) counterpart of:
//   test/ttmlir/Silicon/TTMetal/n150/spatial/reshard_padding_hang_repro.mlir
//
// Root cause (same as the n150 file):
//   When a producer d2m.generic runs on an N-wide core-column grid and a
//   consumer d2m.generic is restricted to an M-wide grid (M < N) via
//   d2m.spatial, the K dimension is padded to ceil(K_tiles / M) * M.  The
//   DMA reshard issues NoC reads for phantom padding tiles that map to source
//   column N (out of bounds for a 0..N-1 grid).
//
// Why the HANG only appears with fabric enabled:
//   - FABRIC DISABLED (n150, single device):
//       The OOB NoC address maps to an ETH core in dormant mode.  The read
//       returns garbage data immediately.  Program completes with WRONG OUTPUT.
//
//   - FABRIC_1D / FABRIC_1D_RING (n300 or any multi-chip host):
//       ETH cores are in ring-forwarding mode.  The phantom NoC read packet
//       is forwarded hop-by-hop around the ring.  No endpoint can serve it,
//       so async_read_barrier never gets its acknowledgment -> INFINITE HANG.
//
// Setup:
//   Producer exp: full device grid (8x8), K = 128 tiles (4096 / 32)
//   Consumer matmul: restricted to 8x7 via d2m.spatial core_range<(0,0),(7,6)>
//   128 % 7 != 0  ->  K padded to ceil(128/7)*7 = 133 tiles
//   Phantom tiles 128..132 map to source column 8 (OOB)
//
// The RUN lines below only COMPILE the MLIR to a flatbuffer (FileCheck pass).
// They deliberately do NOT execute the flatbuffer on device because that
// execution would hang the hardware.
//
// To manually reproduce the hang:
//   1. Compile to flatbuffer with the RUN lines (or ttmlir-opt + ttmlir-translate).
//   2. On an N300 or TG host, run: ttrt run repro.ttm --fabric-config fabric_1d_ring
//   3. The device will hang at async_read_barrier; kill the process and reset
//      with: tt-smi reset
//
// RUN: ttmlir-opt --ttir-to-ttmetal-pipeline="system-desc-path=%system_desc_path% ttnn-mode=false mesh-shape=1,2" -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttmetal-to-flatbuffer -o %t.ttm %t.mlir
//
// CHECK: "ttmetal.enqueue_program"
// CHECK: core_range<0x0, 8x8>
// CHECK: core_range<0x0, 8x7>

// Layouts: base (8x8 compatible, 32x32 alignment)
//          K-padded (7x8 compatible, 224x32 alignment = 7*32 x 32)
#layout   = #ttcore.metal_layout<logical_shape = 256x4096,  dim_alignments = 32x32,  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layoutK  = #ttcore.metal_layout<logical_shape = 256x4096,  dim_alignments = 32x224, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layout1K = #ttcore.metal_layout<logical_shape = 4096x256,  dim_alignments = 224x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#layoutO  = #ttcore.metal_layout<logical_shape = 256x256,   dim_alignments = 32x32,  collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map  = affine_map<(d0, d1) -> (d0, d1)>
#map1 = affine_map<(d0, d1, d2) -> (d0, d2)>
#map2 = affine_map<(d0, d1, d2) -> (d2, d1)>
#map3 = affine_map<(d0, d1, d2) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#reduction = #ttcore.iterator_type<reduction>

module {
  // exp(act) @ weight  where act=[256,4096] weight=[4096,256].
  // exp runs on the full 8x8 grid; matmul is restricted to 8x7.
  func.func @reshard_padding_repro(%arg0: tensor<256x4096xbf16>, %arg1: tensor<4096x256xbf16>) -> tensor<256x256xbf16> attributes {tt.function_type = "forward_device"} {
    // Host -> device: activation (256x4096), 8x8 grid, K=128 tiles.
    %a0 = d2m.empty() : tensor<1x1x8x128x!ttcore.tile<32x32, bf16>, #layout>
    %a1 = d2m.to_layout %arg0, %a0 : tensor<256x4096xbf16> into tensor<1x1x8x128x!ttcore.tile<32x32, bf16>, #layout> -> tensor<1x1x8x128x!ttcore.tile<32x32, bf16>, #layout>

    // Producer: elementwise exp on the FULL device grid (8x8).
    %pe = d2m.empty() : tensor<1x1x8x128x!ttcore.tile<32x32, bf16>, #layout>
    %p = d2m.generic {block_factors = [1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map, #map], iterator_types = [#parallel, #parallel], threads = [#d2m.thread<unified>]}
        ins(%a1 : tensor<1x1x8x128x!ttcore.tile<32x32, bf16>, #layout>)
        outs(%pe : tensor<1x1x8x128x!ttcore.tile<32x32, bf16>, #layout>) {
      %b0 = d2m.block_index(0) : index
      %b1 = d2m.block_index(1) : index
      %t0 = tensor.empty() : tensor<8x128x!ttcore.tile<32x32, bf16>>
      %ld = d2m.remote_load %t0 %a1[%b0, %b1] : tensor<8x128x!ttcore.tile<32x32, bf16>>, tensor<1x1x8x128x!ttcore.tile<32x32, bf16>, #layout> -> tensor<8x128x!ttcore.tile<32x32, bf16>>
      %t1 = tensor.empty() : tensor<8x128x!ttcore.tile<32x32, bf16>>
      %ev = linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%ld : tensor<8x128x!ttcore.tile<32x32, bf16>>) outs(%t1 : tensor<8x128x!ttcore.tile<32x32, bf16>>) {
      ^bb0(%in: !ttcore.tile<32x32, bf16>, %out: !ttcore.tile<32x32, bf16>):
        %e = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
        linalg.yield %e : !ttcore.tile<32x32, bf16>
      } -> tensor<8x128x!ttcore.tile<32x32, bf16>>
      %sb0 = d2m.block_index(0) : index
      %sb1 = d2m.block_index(1) : index
      %st = d2m.remote_store %pe[%sb0, %sb1] %ev : tensor<1x1x8x128x!ttcore.tile<32x32, bf16>, #layout>, tensor<8x128x!ttcore.tile<32x32, bf16>> -> tensor<1x1x8x128x!ttcore.tile<32x32, bf16>, #layout>
      d2m.yield %st : (tensor<1x1x8x128x!ttcore.tile<32x32, bf16>, #layout>)
    } : tensor<1x1x8x128x!ttcore.tile<32x32, bf16>, #layout>

    // Device-to-device reshard: 8-wide (128 K-tiles) -> 7-wide (133 K-tiles).
    // THIS is the problematic reshard. Grid-aware alignment padded K from
    // 128 to 133 tiles to be divisible by 7. The DMA reads tiles 128-132
    // from source column 8, which does not exist (source is only 8 columns
    // wide, indexed 0-7). With ring fabric, the OOB NoC read is forwarded
    // by the ETH core to the next chip in the ring and never acknowledged.
    %re = d2m.empty() : tensor<1x1x8x133x!ttcore.tile<32x32, bf16>, #layoutK>
    %resharded = d2m.to_layout %p, %re : tensor<1x1x8x128x!ttcore.tile<32x32, bf16>, #layout> into tensor<1x1x8x133x!ttcore.tile<32x32, bf16>, #layoutK> -> tensor<1x1x8x133x!ttcore.tile<32x32, bf16>, #layoutK>

    // Host -> device: weight (4096x256), K-rows padded to 133 tiles.
    %we = d2m.empty() : tensor<1x1x133x8x!ttcore.tile<32x32, bf16>, #layout1K>
    %w = d2m.to_layout %arg1, %we : tensor<4096x256xbf16> into tensor<1x1x133x8x!ttcore.tile<32x32, bf16>, #layout1K> -> tensor<1x1x133x8x!ttcore.tile<32x32, bf16>, #layout1K>

    // Consumer: matmul RESTRICTED to 8x7 grid via single-region d2m.spatial.
    %oe = d2m.empty() : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layoutO>
    %c = d2m.spatial {grid_ranges = [#ttcore.core_range<(0, 0), (7, 6)>]}
        ins(%resharded, %w : tensor<1x1x8x133x!ttcore.tile<32x32, bf16>, #layoutK>, tensor<1x1x133x8x!ttcore.tile<32x32, bf16>, #layout1K>)
        outs(%oe : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layoutO>) {
      ^region0():
        %g = d2m.generic {block_factors = [1, 1, 1], grid = #ttcore.grid<1x1>, indexing_maps = [#map1, #map2, #map3], iterator_types = [#parallel, #parallel, #reduction], threads = [#d2m.thread<unified>]}
            ins(%resharded, %w : tensor<1x1x8x133x!ttcore.tile<32x32, bf16>, #layoutK>, tensor<1x1x133x8x!ttcore.tile<32x32, bf16>, #layout1K>)
            outs(%oe : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layoutO>) {
          %block0 = d2m.block_index(0) : index
          %block1 = d2m.block_index(1) : index
          %block2 = d2m.block_index(2) : index
          %block1_1 = d2m.block_index(1) : index
          %block2_2 = d2m.block_index(2) : index
          %t2 = tensor.empty() : tensor<8x133x!ttcore.tile<32x32, bf16>>
          %c0 = arith.constant 0 : index
          %ld0 = d2m.remote_load %t2 %resharded[%block0, %block2] mcast[%c0] : tensor<8x133x!ttcore.tile<32x32, bf16>>, tensor<1x1x8x133x!ttcore.tile<32x32, bf16>, #layoutK> -> tensor<8x133x!ttcore.tile<32x32, bf16>>
          %t3 = tensor.empty() : tensor<133x8x!ttcore.tile<32x32, bf16>>
          %c1 = arith.constant 1 : index
          %ld1 = d2m.remote_load %t3 %w[%block2_2, %block1_1] mcast[%c1] : tensor<133x8x!ttcore.tile<32x32, bf16>>, tensor<1x1x133x8x!ttcore.tile<32x32, bf16>, #layout1K> -> tensor<133x8x!ttcore.tile<32x32, bf16>>
          %t4 = tensor.empty() : tensor<8x8x!ttcore.tile<32x32, bf16>>
          %mm = linalg.generic {indexing_maps = [#map1, #map2, #map3], iterator_types = ["parallel", "parallel", "reduction"]} ins(%ld0, %ld1 : tensor<8x133x!ttcore.tile<32x32, bf16>>, tensor<133x8x!ttcore.tile<32x32, bf16>>) outs(%t4 : tensor<8x8x!ttcore.tile<32x32, bf16>>) {
          ^bb0(%in: !ttcore.tile<32x32, bf16>, %in_w: !ttcore.tile<32x32, bf16>, %acc: !ttcore.tile<32x32, bf16>):
            %25 = "d2m.tile_matmul"(%in, %in_w, %acc) : (!ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>, !ttcore.tile<32x32, bf16>) -> !ttcore.tile<32x32, bf16>
            linalg.yield %25 : !ttcore.tile<32x32, bf16>
          } -> tensor<8x8x!ttcore.tile<32x32, bf16>>
          %sb0 = d2m.block_index(0) : index
          %sb1 = d2m.block_index(1) : index
          %st = d2m.remote_store %oe[%sb0, %sb1] %mm : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layoutO>, tensor<8x8x!ttcore.tile<32x32, bf16>> -> tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layoutO>
          d2m.yield %st : (tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layoutO>)
        } : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layoutO>
        d2m.spatial_yield %g : (tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layoutO>)
    } : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layoutO>

    %rete = d2m.empty() : tensor<256x256xbf16>
    %ret = d2m.to_layout %c, %rete : tensor<1x1x8x8x!ttcore.tile<32x32, bf16>, #layoutO> into tensor<256x256xbf16> -> tensor<256x256xbf16>
    return %ret : tensor<256x256xbf16>
  }
}
