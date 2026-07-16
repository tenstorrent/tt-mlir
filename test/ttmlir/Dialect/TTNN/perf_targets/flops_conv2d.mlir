// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: ttmlir-opt --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/out.json ttnn-perf-metrics-verbose-output-enabled=true" %s -o /dev/null
// RUN: cat %t.dir/out.json | FileCheck %s

// Per-op FLOP accounting for a single conv2d.
// FLOPs are computed from logical attributes (not the padded weight tensor):
//   in_channels=3, groups=1, kernel=3x3  -> macs/out = (3/1)*3*3 = 27
//   output scalars = 1*1*10000*7                     = 70,000
//   flops = 2 * 70,000 * 27                          = 3,780,000
// The op's compute_config carries math_fidelity = hifi4, which must be
// reflected per-op rather than the HiFi2 default.

#dram = #ttnn.buffer_type<dram>
#sysmem = #ttnn.buffer_type<system_memory>
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

#l_in  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 10000 + d1 * 10000 + d2, d3), <1x1>, memref<10000x3xbf16, #dram>, <interleaved>>
#l_w   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 9 + d1 * 3 + d2, d3), <1x1>, memref<63x3xbf16, #sysmem>>
#l_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 10016 + d1 * 10016 + d2, d3), <1x1>, memref<313x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {ttcore.system_desc = #system_desc} {
  func.func @forward(
      %input:  tensor<1x1x10000x3xbf16, #l_in> {ttcore.argument_type = #ttcore.argument_type<input>},
      %weight: tensor<7x3x3x3xbf16, #l_w>       {ttcore.argument_type = #ttcore.argument_type<parameter>}
  ) -> tensor<1x1x10000x7xbf16, #l_out>
      attributes {tt.function_type = "forward_device"} {
    %d = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %0 = "ttnn.conv2d"(%input, %weight, %d) <{
        batch_size = 1 : i32,
        compute_config = #ttnn.device_compute_kernel_config<math_fidelity = hifi4, fp32_dest_acc_en = true>,
        dilation = array<i32: 1, 1>,
        groups = 1 : i32,
        in_channels = 3 : i32,
        input_height = 100 : i32,
        input_width = 100 : i32,
        kernel_size = array<i32: 3, 3>,
        out_channels = 7 : i32,
        padding = array<i32: 1, 1, 1, 1>,
        stride = array<i32: 1, 1>
      }> : (tensor<1x1x10000x3xbf16, #l_in>, tensor<7x3x3x3xbf16, #l_w>, !ttnn.device) -> tensor<1x1x10000x7xbf16, #l_out>
    return %0 : tensor<1x1x10000x7xbf16, #l_out>
  }
}

// CHECK:      "flops": {
// CHECK:        "flops_by_category": {
// CHECK:          "conv": 3780000
// CHECK:        }
// CHECK:        "per_op": [
// CHECK:          "category": "conv"
// CHECK:          "flops": 3780000
// CHECK:          "math_fidelity": "hifi4"
// CHECK:          "operation": "ttnn.conv2d"
// CHECK:        "total_flops": 3780000
