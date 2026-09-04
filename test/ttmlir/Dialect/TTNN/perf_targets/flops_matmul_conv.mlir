// RUN: rm -rf %t.dir && mkdir -p %t.dir
// RUN: ttmlir-opt --ttnn-collect-perf-metrics="ttnn-perf-metrics-output-file=%t.dir/out.json ttnn-perf-metrics-verbose-output-enabled=true" %s -o /dev/null
// RUN: cat %t.dir/out.json | FileCheck %s

// Matmul and conv FLOP formulas. Every counted op sets math_fidelity = lofi, so
// peak_flops_per_sec must come out as the lofi peak rather than the HiFi4
// default.
//
//   matmul    M=2048, K=128, N=128           -> 2*M*K*N          = 67,108,864
//   matmul    transpose_a, A=[64,512], B=[64,32] -> [512,32]:
//             K is A[-2]=64                  -> 2*512*64*32      =  2,097,152
//   conv2d    in_ch=8, groups=4, kernel 3x3, out scalars 1024*8:
//             2*8192*(8/4)*9                                     =    294,912
//   conv_t2d  stride 2, in scalars 256*8, out_ch=8, kernel 2x2:
//             2*2048*8*4                                         =    131,072
//             Input-driven; the output volume would give 524,288 = 4x = stride^2.
//   relu/add  not counted
//   total                                                        = 69,632,000
//
// lofi peak (WH B0, 64 cores, 1.0 GHz) = 64 * 1e9 * 2*32^3 / 16 = 2.62144e14.

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

// matmul
#mm_a = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#mm_b = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<4x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// matmul with transpose_a
#ta_a = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x16x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ta_b = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ta_o = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<16x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
// convs: flattened (1, 1, N*spatial, C) activations, (O, C/G, KH, KW) weights
#c_in  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<1024x8xbf16, #dram>, <interleaved>>
#c_w   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 6 + d1 * 3 + d2, d3), <1x1>, memref<48x3xbf16, #sysmem>>
#c_out = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 1024 + d2, d3), <1x1>, memref<32x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#t_in  = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 * 256 + d2, d3), <1x1>, memref<256x8xbf16, #dram>, <interleaved>>
#t_w   = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16 + d1 * 2 + d2, d3), <1x1>, memref<128x2xbf16, #sysmem>>

module attributes {ttcore.system_desc = #system_desc} {
  func.func @forward(
      %a: tensor<2048x128xbf16, #mm_a>      {ttcore.argument_type = #ttcore.argument_type<input>},
      %b: tensor<128x128xbf16, #mm_b>       {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %ta: tensor<64x512xbf16, #ta_a>       {ttcore.argument_type = #ttcore.argument_type<input>},
      %tb: tensor<64x32xbf16, #ta_b>        {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %ci: tensor<1x1x1024x8xbf16, #c_in>   {ttcore.argument_type = #ttcore.argument_type<input>},
      %cw: tensor<8x2x3x3xbf16, #c_w>       {ttcore.argument_type = #ttcore.argument_type<parameter>},
      %ti: tensor<1x1x256x8xbf16, #t_in>    {ttcore.argument_type = #ttcore.argument_type<input>},
      %tw: tensor<8x8x2x2xbf16, #t_w>       {ttcore.argument_type = #ttcore.argument_type<parameter>}
  ) -> tensor<1x1x1024x8xbf16, #c_out>
      attributes {tt.function_type = "forward_device"} {
    %d = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device

    %0 = "ttnn.matmul"(%a, %b) <{
        transpose_a = false, transpose_b = false,
        compute_config = #ttnn.device_compute_kernel_config<math_fidelity = lofi>
      }> : (tensor<2048x128xbf16, #mm_a>, tensor<128x128xbf16, #mm_b>) -> tensor<2048x128xbf16, #mm_a>

    // Not counted.
    %1 = "ttnn.relu"(%0) : (tensor<2048x128xbf16, #mm_a>) -> tensor<2048x128xbf16, #mm_a>
    %2 = "ttnn.add"(%1, %0) <{activations = [], input_tensor_a_activations = [], input_tensor_b_activations = []}> : (tensor<2048x128xbf16, #mm_a>, tensor<2048x128xbf16, #mm_a>) -> tensor<2048x128xbf16, #mm_a>

    // transpose_a: K comes from A[-2] = 64.
    %3 = "ttnn.matmul"(%ta, %tb) <{
        transpose_a = true, transpose_b = false,
        compute_config = #ttnn.device_compute_kernel_config<math_fidelity = lofi>
      }> : (tensor<64x512xbf16, #ta_a>, tensor<64x32xbf16, #ta_b>) -> tensor<512x32xbf16, #ta_o>

    %4 = "ttnn.conv2d"(%ci, %cw, %d) <{
        batch_size = 1 : i32,
        compute_config = #ttnn.device_compute_kernel_config<math_fidelity = lofi>,
        dilation = array<i32: 1, 1>,
        groups = 4 : i32,
        in_channels = 8 : i32,
        input_height = 32 : i32,
        input_width = 32 : i32,
        kernel_size = array<i32: 3, 3>,
        out_channels = 8 : i32,
        padding = array<i32: 1, 1, 1, 1>,
        stride = array<i32: 1, 1>
      }> : (tensor<1x1x1024x8xbf16, #c_in>, tensor<8x2x3x3xbf16, #c_w>, !ttnn.device) -> tensor<1x1x1024x8xbf16, #c_out>

    %5 = "ttnn.conv_transpose2d"(%ti, %tw, %d) <{
        batch_size = 1 : i32,
        compute_config = #ttnn.device_compute_kernel_config<math_fidelity = lofi>,
        dilation = array<i32: 1, 1>,
        groups = 1 : i32,
        in_channels = 8 : i32,
        input_height = 16 : i32,
        input_width = 16 : i32,
        kernel_size = array<i32: 2, 2>,
        out_channels = 8 : i32,
        output_padding = array<i32: 0, 0>,
        padding = array<i32: 0, 0>,
        stride = array<i32: 2, 2>
      }> : (tensor<1x1x256x8xbf16, #t_in>, tensor<8x8x2x2xbf16, #t_w>, !ttnn.device) -> tensor<1x1x1024x8xbf16, #c_out>

    return %5 : tensor<1x1x1024x8xbf16, #c_out>
  }
}

// Keys are emitted alphabetically.
// CHECK:      "flops": {
// CHECK:        "peak_flops_per_sec": 262144000000000
// CHECK:        "peak_flops_per_sec_by_fidelity": {
// CHECK:          "hifi2": 131072000000000
// CHECK:          "hifi3": 87381333333333
// CHECK:          "hifi4": 65536000000000
// CHECK:          "lofi": 262144000000000
// CHECK:        }
// CHECK:        "per_op": [
// CHECK:          "flops": 67108864
// CHECK:          "math_fidelity": "lofi"
// CHECK:          "operation": "ttnn.matmul"
// CHECK-NOT:      "ttnn.relu"
// CHECK-NOT:      "ttnn.add"
// CHECK:          "flops": 2097152
// CHECK:          "operation": "ttnn.matmul"
// CHECK:          "flops": 294912
// CHECK:          "operation": "ttnn.conv2d"
// CHECK:          "flops": 131072
// CHECK:          "operation": "ttnn.conv_transpose2d"
// CHECK:        "total_flops": 69632000
