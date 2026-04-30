// RUN: ttmlir-opt --convert-d2m-to-ttnn -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// DRAM interleaved, shape (64, 128), unary abs with streaming CBs. Exercises
// --convert-d2m-to-ttnn on IR extracted from the JIT pipeline.

// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.dealloc
// CHECK-NOT: d2m.view_layout
// CHECK-NOT: d2m.generic
// CHECK-NOT: d2m.empty

// CHECK: "ttnn.get_device"
// CHECK: %[[OUT:.*]] = "ttnn.empty"{{.*}}<interleaved>{{.*}}shape = #ttnn.shape<64x128>
// CHECK: "ttnn.generic"(%arg0, %[[OUT]])
// CHECK-SAME: operandSegmentSizes = array<i32: 2, 0>
// CHECK-SAME: symbol_ref = @datamovement_kernel0, core_ranges = <[#ttnn.core_range<(0,0), (3,1)>]>, processor = riscv1, noc_index = noc0, noc_mode = dedicated_noc, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>], common_rt_args = [#ttnn.kernel_arg_address_of_tensor<0>], rt_args = []
// CHECK-SAME: symbol_ref = @datamovement_kernel1, core_ranges = <[#ttnn.core_range<(0,0), (3,1)>]>, processor = riscv0, noc_index = noc1, noc_mode = dedicated_noc, ct_args = [#ttnn.kernel_arg_cb_buffer_index<1>], common_rt_args = [#ttnn.kernel_arg_address_of_tensor<1>], rt_args = []
// CHECK-SAME: symbol_ref = @compute_kernel2, core_ranges = <[#ttnn.core_range<(0,0), (3,1)>]>, math_fidelity = hifi4, fp32_dest_acc_en = false, dst_full_sync_en = false, unpack_to_dest_modes = [default], bfp8_pack_precise = false, math_approx_mode = false, ct_args = [#ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<0>], common_rt_args = [], rt_args = []
// CHECK-SAME: <total_size = 8192, core_ranges = <[#ttnn.core_range<(0,0), (3,1)>]>, formats = [<buffer_index = 0, dtype = f32, page_size = 4096>]>
// CHECK-SAME: <total_size = 8192, core_ranges = <[#ttnn.core_range<(0,0), (3,1)>]>, formats = [<buffer_index = 1, dtype = f32, page_size = 4096>]>
// CHECK-SAME: semaphores = []>
// CHECK: return %[[OUT]]

module attributes {ttcore.system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073119552, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2, dram_grid = 1x12, dram_bank_to_logical_worker_noc0 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)], dram_bank_to_logical_worker_noc1 = [(7, 3), (0, 0), (3, 0), (4, 0), (0, 4), (7, 7), (1, 4), (6, 4), (5, 4), (2, 6), (3, 4), (4, 4)]}], [0], [1 : i32], [ 0x0x0x0]>} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @abs_test(%arg0: tensor<64x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>, exactGrid = true>>) -> tensor<64x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>, exactGrid = true>> attributes {tt.function_type = "forward_device"} {
    %0 = d2m.empty() : tensor<64x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>, exactGrid = true>>
    %cast = ttir.ttnn_metal_layout_cast %arg0 : tensor<64x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>, exactGrid = true>> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #ttcore.memory_space<dram>>
    %cast_0 = ttir.ttnn_metal_layout_cast %0 : tensor<64x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>, exactGrid = true>> -> memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #ttcore.memory_space<dram>>
    %view = d2m.view_layout %cast_0 remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, (d0 + d1 floordiv 4) mod 2, d1 mod 4)> : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #ttcore.memory_space<dram>> -> memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<dram>>
    %view_1 = d2m.view_layout %cast remapping = affine_map<(d0, d1, d2, d3) -> (0, 0, (d0 + d1 floordiv 4) mod 2, d1 mod 4)> : memref<1x1x2x4x!ttcore.tile<32x32, f32>, #ttcore.interleaved<16384x4096>, #ttcore.memory_space<dram>> -> memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<dram>>
    %alloc_3 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #ttcore.memory_space<l1>>
    %alloc_4 = memref.alloc() {address = 111904 : i64, alignment = 16 : i64} : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [], grid = #ttcore.grid<2x4>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, @datamovement_kernel0, noc = 0>, #d2m.thread<datamovement, @datamovement_kernel1, noc = 1>, #d2m.thread<compute, @compute_kernel2>]}
        ins(%view_1 : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<dram>>)
        outs(%view : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<dram>>)
        additionalArgs(%alloc_3, %alloc_4 : memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #ttcore.memory_space<l1>>, memref<1x1x!ttcore.tile<32x32, f32>, #ttcore.cb_layout<4096x4096, 2>, #ttcore.memory_space<l1>>)

    %cast_5 = ttir.ttnn_metal_layout_cast %view : memref<2x4x1x1x!ttcore.tile<32x32, f32>, #ttcore.view<4>, #ttcore.memory_space<dram>> -> tensor<64x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>, exactGrid = true>>
    return %cast_5 : tensor<64x128xf32, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>, exactGrid = true>>
  }
  func.func private @datamovement_kernel0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @datamovement_kernel1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 3>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @compute_kernel2() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 3>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}
