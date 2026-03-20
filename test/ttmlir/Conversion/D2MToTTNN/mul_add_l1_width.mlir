// RUN: ttmlir-opt --convert-d2m-to-ttnn -o %t.mlir %s
// RUN: FileCheck %s --input-file=%t.mlir

// L1 width-sharded, shape (128, 2048), mixed streaming + aliased. Exercises only --convert-d2m-to-ttnn on IR extracted from the JIT pipeline.

// CHECK-NOT: memref.alloc
// CHECK-NOT: memref.dealloc
// CHECK-NOT: d2m.stream_layout
// CHECK-NOT: d2m.view_layout
// CHECK-NOT: d2m.generic
// CHECK-NOT: d2m.empty

// CHECK: "ttnn.get_device"
// CHECK: %[[MUL_OUT:.*]] = "ttnn.empty"{{.*}}<block_sharded>{{.*}}core_range<(0,0), (7,3)>{{.*}}shape = #ttnn.shape<128x2048>
// CHECK: "ttnn.generic"(%arg1, %arg2, %[[MUL_OUT]])
// CHECK-SAME: operandSegmentSizes = array<i32: 3, 0>
// CHECK-SAME: symbol_ref = @datamovement_kernel0, core_ranges = <[#ttnn.core_range<(0,0), (7,3)>]>, processor = riscv1, noc_index = noc0, noc_mode = dedicated_noc, ct_args = [#ttnn.kernel_arg_cb_buffer_index<0>], common_rt_args = [#ttnn.kernel_arg_address_of_tensor<0>], rt_args = []
// CHECK-SAME: symbol_ref = @datamovement_kernel1, core_ranges = <[#ttnn.core_range<(0,0), (7,3)>]>, processor = riscv0, noc_index = noc1, noc_mode = dedicated_noc, ct_args = [#ttnn.kernel_arg_cb_buffer_index<1>], common_rt_args = [#ttnn.kernel_arg_address_of_tensor<1>], rt_args = []
// CHECK-SAME: symbol_ref = @compute_kernel2, core_ranges = <[#ttnn.core_range<(0,0), (7,3)>]>, math_fidelity = hifi4, fp32_dest_acc_en = false, dst_full_sync_en = false, unpack_to_dest_modes = [default], bfp8_pack_precise = false, math_approx_mode = false, ct_args = [#ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<0>, #ttnn.kernel_arg_cb_buffer_index<2>], common_rt_args = [], rt_args = []
// CHECK-SAME: <total_size = 32768, core_ranges = <[#ttnn.core_range<(0,0), (7,3)>]>, formats = [<buffer_index = 0, dtype = bf16, page_size = 2048>]>
// CHECK-SAME: <total_size = 32768, core_ranges = <[#ttnn.core_range<(0,0), (7,3)>]>, formats = [<buffer_index = 1, dtype = bf16, page_size = 2048>]>
// CHECK-SAME: <total_size = 16384, core_ranges = <[#ttnn.core_range<(0,0), (7,3)>]>, formats = [<buffer_index = 2, dtype = bf16, page_size = 2048>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<2>>
// CHECK-SAME: semaphores = []>
// CHECK: %[[ADD_OUT:.*]] = "ttnn.empty"{{.*}}<block_sharded>{{.*}}core_range<(0,0), (7,3)>{{.*}}shape = #ttnn.shape<128x2048>
// CHECK: "ttnn.generic"(%[[MUL_OUT]], %arg0, %[[ADD_OUT]])
// CHECK-SAME: operandSegmentSizes = array<i32: 3, 0>
// CHECK-SAME: symbol_ref = @datamovement_kernel3, core_ranges = <[#ttnn.core_range<(0,0), (7,3)>]>, processor = riscv1, noc_index = noc0, noc_mode = dedicated_noc, ct_args = [#ttnn.kernel_arg_cb_buffer_index<1>], common_rt_args = [#ttnn.kernel_arg_address_of_tensor<1>], rt_args = []
// CHECK-SAME: symbol_ref = @compute_kernel4, core_ranges = <[#ttnn.core_range<(0,0), (7,3)>]>, math_fidelity = hifi4, fp32_dest_acc_en = false, dst_full_sync_en = false, unpack_to_dest_modes = [default], bfp8_pack_precise = false, math_approx_mode = false, ct_args = [#ttnn.kernel_arg_cb_buffer_index<1>, #ttnn.kernel_arg_cb_buffer_index<2>, #ttnn.kernel_arg_cb_buffer_index<0>], common_rt_args = [], rt_args = []
// CHECK-SAME: <total_size = 16384, core_ranges = <[#ttnn.core_range<(0,0), (7,3)>]>, formats = [<buffer_index = 0, dtype = bf16, page_size = 2048>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<0>>
// CHECK-SAME: <total_size = 32768, core_ranges = <[#ttnn.core_range<(0,0), (7,3)>]>, formats = [<buffer_index = 1, dtype = bf16, page_size = 2048>]>
// CHECK-SAME: <total_size = 16384, core_ranges = <[#ttnn.core_range<(0,0), (7,3)>]>, formats = [<buffer_index = 2, dtype = bf16, page_size = 2048>], buffer = #ttnn.kernel_cb_global_buffer_address_of_tensor<2>>
// CHECK-SAME: semaphores = []>
// CHECK: return %[[ADD_OUT]]

module attributes {ttcore.system_desc = #ttcore.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, coord_translation_offsets = 18x18, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 103712, erisc_l1_unreserved_base = 98304, dram_unreserved_base = 1920032, dram_unreserved_end = 1073119552, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>, <si32>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], dst_physical_size_tiles = 16, num_cbs = 64, num_compute_threads = 1, num_datamovement_threads = 2}], [0], [1 : i32], [ 0x0x0x0]>} {
  ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, (((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) mod 12, ((((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) floordiv s4) floordiv 12) * s4 + ((d0 * s1) * (s2 * (s3 * s6)) + d1 * (s2 * (s3 * s6)) + d2) mod s4 + s5), meshShape = , chipIds = [0]>
  func.func @mul_add(%arg0: tensor<128x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<4x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <width_sharded>, exactGrid = true>>, %arg1: tensor<128x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<4x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <width_sharded>, exactGrid = true>>, %arg2: tensor<128x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<4x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <width_sharded>, exactGrid = true>>) -> tensor<128x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <4x8>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>, exactGrid = true>> attributes {tt.function_type = "forward_device"} {
    %cast = ttir.ttnn_metal_layout_cast %arg1 {virtual_grid_forward_mapping = affine_map<(d0, d1, d2, d3) -> ((d1 floordiv 8) mod 8, d1 mod 8, d2, d3)>, virtual_grid_inverse_mapping = affine_map<(d0, d1) -> (0, 0, (d1 + d0 * 8) mod 64)>} : tensor<128x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<4x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <width_sharded>, exactGrid = true>> -> memref<1x64x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>
    %cast_0 = ttir.ttnn_metal_layout_cast %arg2 {virtual_grid_forward_mapping = affine_map<(d0, d1, d2, d3) -> ((d1 floordiv 8) mod 8, d1 mod 8, d2, d3)>, virtual_grid_inverse_mapping = affine_map<(d0, d1) -> (0, 0, (d1 + d0 * 8) mod 64)>} : tensor<128x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<4x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <width_sharded>, exactGrid = true>> -> memref<1x64x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>
    %alloc = memref.alloc() {address = 169248 : i64, alignment = 16 : i64} : memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 1>, #ttcore.memory_space<l1>>
    %alloc_1 = memref.alloc() : memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 2>, #ttcore.memory_space<l1>>
    %stream = "d2m.stream_layout"(%cast, %alloc_1) <{remapping = affine_map<(d0, d1, d2, d3) -> (0, (d1 * 8 + d3) mod 64, (d0 + (d1 * 8 + d3) floordiv 64) mod 4, 0)>}> : (memref<1x64x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>, memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 2>, #ttcore.memory_space<l1>>) -> memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.view<4>, #ttcore.memory_space<l1>>
    %alloc_2 = memref.alloc() : memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 2>, #ttcore.memory_space<l1>>
    %stream_3 = "d2m.stream_layout"(%cast_0, %alloc_2) <{remapping = affine_map<(d0, d1, d2, d3) -> (0, (d1 * 8 + d3) mod 64, (d0 + (d1 * 8 + d3) floordiv 64) mod 4, 0)>}> : (memref<1x64x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>, memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 2>, #ttcore.memory_space<l1>>) -> memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.view<4>, #ttcore.memory_space<l1>>
    %alloc_4 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64, d2m.cb_for_operand = 0 : i64} : memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<16384x2048, 2, grid = [4x8]>, #ttcore.memory_space<l1>>
    %alloc_5 = memref.alloc() {address = 136480 : i64, alignment = 16 : i64, d2m.cb_for_operand = 1 : i64} : memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<16384x2048, 2, grid = [4x8]>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x8>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, noc = 0, @datamovement_kernel0>, #d2m.thread<datamovement, noc = 1, @datamovement_kernel1>, #d2m.thread<compute, noc = 0, @compute_kernel2>]}
        ins(%stream, %stream_3 : memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.view<4>, #ttcore.memory_space<l1>>, memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.view<4>, #ttcore.memory_space<l1>>)
        outs(%alloc : memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 1>, #ttcore.memory_space<l1>>)
        additionalArgs(%alloc_4, %alloc_5 : memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<16384x2048, 2, grid = [4x8]>, #ttcore.memory_space<l1>>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<16384x2048, 2, grid = [4x8]>, #ttcore.memory_space<l1>>)

    %cast_6 = ttir.ttnn_metal_layout_cast %arg0 {virtual_grid_forward_mapping = affine_map<(d0, d1, d2, d3) -> ((d1 floordiv 8) mod 8, d1 mod 8, d2, d3)>, virtual_grid_inverse_mapping = affine_map<(d0, d1) -> (0, 0, (d1 + d0 * 8) mod 64)>} : tensor<128x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <8x8>, memref<4x1x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <width_sharded>, exactGrid = true>> -> memref<1x64x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>
    %alloc_7 = memref.alloc() {address = 136480 : i64, alignment = 16 : i64} : memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 1>, #ttcore.memory_space<l1>>
    %alloc_8 = memref.alloc() : memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 2>, #ttcore.memory_space<l1>>
    %stream_9 = "d2m.stream_layout"(%cast_6, %alloc_8) <{remapping = affine_map<(d0, d1, d2, d3) -> (0, (d1 * 8 + d3) mod 64, (d0 + (d1 * 8 + d3) floordiv 64) mod 4, 0)>}> : (memref<1x64x4x1x!ttcore.tile<32x32, bf16>, #ttcore.shard<2048x2048, 1>, #ttcore.memory_space<l1>>, memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 2>, #ttcore.memory_space<l1>>) -> memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.view<4>, #ttcore.memory_space<l1>>
    %alloc_10 = memref.alloc() {address = 103712 : i64, alignment = 16 : i64, d2m.cb_for_operand = 1 : i64} : memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<16384x2048, 2, grid = [4x8]>, #ttcore.memory_space<l1>>
    d2m.generic {block_factors = [], grid = #ttcore.grid<4x8>, indexing_maps = [], iterator_types = [], threads = [#d2m.thread<datamovement, noc = 0, @datamovement_kernel3>, #d2m.thread<compute, noc = 0, @compute_kernel4>]}
        ins(%alloc, %stream_9 : memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 1>, #ttcore.memory_space<l1>>, memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.view<4>, #ttcore.memory_space<l1>>)
        outs(%alloc_7 : memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 1>, #ttcore.memory_space<l1>>)
        additionalArgs(%alloc_10 : memref<1x8x!ttcore.tile<32x32, bf16>, #ttcore.cb_layout<16384x2048, 2, grid = [4x8]>, #ttcore.memory_space<l1>>)

    memref.dealloc %alloc : memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 1>, #ttcore.memory_space<l1>>
    %cast_11 = ttir.ttnn_metal_layout_cast %alloc_7 : memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 1>, #ttcore.memory_space<l1>> -> tensor<128x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <4x8>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>, exactGrid = true>>
    memref.dealloc %alloc_7 : memref<4x8x1x8x!ttcore.tile<32x32, bf16>, #ttcore.shard<16384x2048, 1>, #ttcore.memory_space<l1>>
    return %cast_11 : tensor<128x2048xbf16, #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <4x8>, memref<1x8x!ttcore.tile<32x32, bf16>, #ttnn.buffer_type<l1>>, <block_sharded>, exactGrid = true>>
  }
  func.func private @datamovement_kernel0() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 0>] ct_args = [<arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @datamovement_kernel1() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @compute_kernel2() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 0>, <arg_type = cb_port, operand_index = 2>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
  func.func private @datamovement_kernel3() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec<rt_args = [<arg_type = buffer_address, operand_index = 1>] ct_args = [<arg_type = cb_port, operand_index = 1>]>, ttkernel.thread = #ttkernel.thread<noc>} {
    return
  }
  func.func private @compute_kernel4() attributes {tt.function_type = "kernel", ttkernel.arg_spec = #ttkernel.arg_spec< ct_args = [<arg_type = cb_port, operand_index = 1>, <arg_type = cb_port, operand_index = 2>, <arg_type = cb_port, operand_index = 0>]>, ttkernel.thread = #ttkernel.thread<compute>} {
    return
  }
}
