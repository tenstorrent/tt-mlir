#l1 = #ttcore.memory_space<l1>
#layout = #ttcore.metal_layout<logical_shape = 32x32, dim_alignments = 32x32, collapsed_intervals = dense<[[0, 1], [1, 2]]> : tensor<2x2xi64>, l1, sharded>
#map = affine_map<(d0, d1) -> (d0, d1)>
#parallel = #ttcore.iterator_type<parallel>
#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103840, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1048640, dram_unreserved_end = 1073162048, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (4, 0), (1, 2), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 4), (4, 4), (3, 6)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (4, 0), (1, 2), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 4), (4, 4), (3, 6)]}, {arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103840, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1048640, dram_unreserved_end = 1073170688, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 2), (3, 0), (4, 0), (2, 6), (7, 7), (0, 4), (6, 4), (5, 4), (1, 4), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 2), (3, 0), (4, 0), (2, 6), (7, 7), (0, 4), (6, 4), (5, 4), (1, 4), (3, 4), (4, 4)]}], [0, 1], [1 : i32, 0 : i32], [ 0x0x0x0], [<[0, 9, 0], [1, 1, 0]>]>
module attributes {ttcore.system_desc = #system_desc} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @main(%arg0: memref<32x32xf32>, %arg1: index, %arg2: index) -> memref<32x32xf32> attributes {tt.function_type = "forward_device"} {
    %alloc_1 = memref.alloc() : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    d2m.to_device %arg0, %alloc_1 layout = #layout : memref<32x32xf32> into memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    %alloc_2 = memref.alloc() : memref<32x32xf32>
    %alloc_3 = memref.alloc() : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
    d2m.generic {block_factors = [], grid = #ttcore.grid<1x1>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<unified>]}
        ins(%alloc_1 : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>)
        outs(%alloc_3 : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>)
        additionalArgs(%arg1, %arg2 : index, index)
     {
      %cst = arith.constant 0.000000e+00 : f32
      %cst_4 = arith.constant 0xFF800000 : f32
      %cst_5 = arith.constant 1.000000e+00 : f32
      %c0 = arith.constant 0 : index
      %rm = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
      d2m.remote_load %rm %alloc_1[%c0, %c0] : memref<32x32xf32>, memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>
      %tin = memref.alloc() {alignment = 64 : i64} : memref<1x1x!ttcore.tile<32x32, f32>>
      "d2m.tile_tilize_block"(%rm, %tin) : (memref<32x32xf32>, memref<1x1x!ttcore.tile<32x32, f32>>) -> ()
      %alloc_7 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>>
      %alloc_8 = memref.alloc() {d2m.reduction_scaler} : memref<1x1x!ttcore.tile<32x32, f32>>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%alloc_8 : memref<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%out: !ttcore.tile<32x32, f32>):
        %0 = d2m.tile_fill(%cst_5) : f32 -> <32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%tin, %alloc_8 : memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x1x!ttcore.tile<32x32, f32>>) outs(%alloc_7 : memref<1x1x!ttcore.tile<32x32, f32>>) attrs =  {d2m.reduced_axes = [1]} {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_14: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = d2m.tile_fill(%cst_4) : f32 -> <32x32, f32>
        %1 = "d2m.tile_reduce_max"(%in, %in_14, %0) <{reduce_dim = #d2m<reduce_dim R>}> : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %1 : !ttcore.tile<32x32, f32>
      }
      %alloc_9 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%tin, %alloc_7 : memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x1x!ttcore.tile<32x32, f32>>) outs(%alloc_9 : memref<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_14: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_bcast"(%in_14) <{bcast_type = #d2m<tile_bcast_type col>}> : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        %1 = "d2m.tile_sub"(%in, %0) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %1 : !ttcore.tile<32x32, f32>
      }
      %alloc_10 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>>
      linalg.generic {indexing_maps = [#map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_9 : memref<1x1x!ttcore.tile<32x32, f32>>) outs(%alloc_10 : memref<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_exp"(%in) : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
      %alloc_11 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>>
      %alloc_12 = memref.alloc() {d2m.reduction_scaler} : memref<1x1x!ttcore.tile<32x32, f32>>
      linalg.generic {indexing_maps = [#map], iterator_types = ["parallel", "parallel"]} outs(%alloc_12 : memref<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%out: !ttcore.tile<32x32, f32>):
        %0 = d2m.tile_fill(%cst_5) : f32 -> <32x32, f32>
        linalg.yield %0 : !ttcore.tile<32x32, f32>
      }
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_10, %alloc_12 : memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x1x!ttcore.tile<32x32, f32>>) outs(%alloc_11 : memref<1x1x!ttcore.tile<32x32, f32>>) attrs =  {d2m.reduced_axes = [1]} {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_14: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = d2m.tile_fill(%cst) : f32 -> <32x32, f32>
        %1 = "d2m.tile_reduce_sum"(%in, %in_14, %0) <{reduce_dim = #d2m<reduce_dim R>}> : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %1 : !ttcore.tile<32x32, f32>
      }
      %alloc_13 = memref.alloc() : memref<1x1x!ttcore.tile<32x32, f32>>
      linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%alloc_10, %alloc_11 : memref<1x1x!ttcore.tile<32x32, f32>>, memref<1x1x!ttcore.tile<32x32, f32>>) outs(%alloc_13 : memref<1x1x!ttcore.tile<32x32, f32>>) {
      ^bb0(%in: !ttcore.tile<32x32, f32>, %in_14: !ttcore.tile<32x32, f32>, %out: !ttcore.tile<32x32, f32>):
        %0 = "d2m.tile_bcast"(%in_14) <{bcast_type = #d2m<tile_bcast_type col>}> : (!ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        %1 = "d2m.tile_div"(%in, %0) : (!ttcore.tile<32x32, f32>, !ttcore.tile<32x32, f32>) -> !ttcore.tile<32x32, f32>
        linalg.yield %1 : !ttcore.tile<32x32, f32>
      }
      %rmo = memref.alloc() {alignment = 64 : i64} : memref<32x32xf32>
      "d2m.tile_untilize_block"(%alloc_13, %rmo) : (memref<1x1x!ttcore.tile<32x32, f32>>, memref<32x32xf32>) -> ()
      d2m.remote_store %alloc_3[%c0, %c0] %rmo : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1>, memref<32x32xf32>
    }
    d2m.to_host %alloc_3, %alloc_2 layout = #layout : memref<1x1x32x32xf32, #ttcore.shard<128x4, 1>, #l1> into memref<32x32xf32>
    return %alloc_2 : memref<32x32xf32>
  }
}
