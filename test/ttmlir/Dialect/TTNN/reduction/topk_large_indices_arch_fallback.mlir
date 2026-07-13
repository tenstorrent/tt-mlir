// RUN: ttmlir-opt --ttnn-resolve-composites="composite-resolution=force-promote" %s | FileCheck %s

// ttnn.experimental.topk_large_indices is Blackhole-only. On a non-Blackhole
// target the promotion guard vetoes promotion, so TTNNResolveComposites inlines
// the decomposition body instead of promoting -- even under force-promote,
// which would otherwise promote to the typed op. The wormhole system_desc is
// what triggers the inline fallback.

#system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux-gnu"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 1024, erisc_l1_unreserved_base = 1024, dram_unreserved_base = 1024, dram_unreserved_end = 1073741824, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 32, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>

module attributes {ttcore.system_desc = #system_desc} {
  func.func @topk_large_indices(%input: tensor<1x512xbf16>) -> tensor<1x512xui32> {
    // CHECK-LABEL: @topk_large_indices
    // CHECK-NOT: "ttnn.topk_large_indices"
    // CHECK-NOT: "ttcore.composite"
    // The decomposition body (ttir.topk) is spliced in verbatim.
    // CHECK: "ttir.topk"
    %0 = "ttcore.composite"(%input) <{composite_name = "topk_large_indices", decomposition = @decomp, composite_attributes = {k = 512 : ui32}}> : (tensor<1x512xbf16>) -> tensor<1x512xui32>
    return %0 : tensor<1x512xui32>
  }
  func.func private @decomp(%input: tensor<1x512xbf16>) -> tensor<1x512xui32> {
    %values, %indices = "ttir.topk"(%input) <{k = 512 : i32, dim = -1 : i32, largest = true, sorted = true}> : (tensor<1x512xbf16>) -> (tensor<1x512xbf16>, tensor<1x512xui32>)
    return %indices : tensor<1x512xui32>
  }
}
