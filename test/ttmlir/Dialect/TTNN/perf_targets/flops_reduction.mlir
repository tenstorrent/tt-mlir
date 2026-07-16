// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: ttmlir-opt --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/out.json ttnn-perf-metrics-verbose-output-enabled=true" %s -o /dev/null
// RUN: cat %t.dir/out.json | FileCheck %s

// Per-op FLOP accounting for a reduction (one accumulate per input scalar):
//   input 512x512 -> flops = num_input_elements = 512 * 512 = 262,144

#dram = #ttnn.buffer_type<dram>
#system_desc = #ttcore.system_desc<[{
  role = host, target_triple = "x86_64-pc-linux"
}], [{
  arch = <wormhole_b0>,
  grid = 8x8,
  coord_translation_offsets = 18x18,
  l1_size = 1499136,
  num_dram_channels = 12,
  dram_channel_size = 1073741824,
  noc_l1_address_align_bytes = 16,
  pcie_address_align_bytes = 32,
  noc_dram_address_align_bytes = 32,
  l1_unreserved_base = 103712,
  erisc_l1_unreserved_base = 98304,
  dram_unreserved_base = 1920032,
  dram_unreserved_end = 1073119552,
  supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>],
  supported_tile_sizes = [ 4x16, 16x16, 32x16, 4x32, 16x32, 32x32 ],
  dst_physical_size_tiles = 16,
  num_cbs = 64,
  num_compute_threads = 1,
  num_datamovement_threads = 2,
  dram_grid = 1x12,
  dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)],
  dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]
}], [0], [1 : i32], [ 0x0x0x0]>

#in  = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#out = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {ttcore.system_desc = #system_desc} {
  func.func @forward(
      %input: tensor<512x512xbf16, #in> {ttcore.argument_type = #ttcore.argument_type<input>}
  ) -> tensor<512x1xbf16, #out>
      attributes {tt.function_type = "forward_device"} {
    %0 = "ttnn.sum"(%input) <{dim_arg = [1 : i32], keep_dim = true}>
        : (tensor<512x512xbf16, #in>) -> tensor<512x1xbf16, #out>
    return %0 : tensor<512x1xbf16, #out>
  }
}

// CHECK:      "flops": {
// CHECK:        "flops_by_category": {
// CHECK:          "reduction": 262144
// CHECK:        }
// CHECK:        "per_op": [
// CHECK:          "category": "reduction"
// CHECK:          "flops": 262144
// CHECK:          "operation": "ttnn.sum"
// CHECK:        "total_flops": 262144
