// RUN: ttmlir-opt --ttir-load-system-desc="path=%system_desc_path%" --ttir-implicit-device="force-reload=true" --ttnn-decompose-layouts %s > %t.mlir
// RUN: FileCheck %s --input-file=%t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn
#device = #tt.device<workerGrid = #tt.grid<8x8, (d0, d1) -> (0, d0, d1)>, l1Map = (d0, d1)[s0, s1] -> (0, d0 floordiv s0, d1 floordiv s1, (d0 mod s0) * s1 + d1 mod s1), dramMap = (d0, d1)[s0, s1] -> (0, 0, ((((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 8192) mod 12, (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) floordiv 98304 + (((d0 floordiv s0) * 8 + d1 floordiv s1) * (s1 * s0) + (d0 mod s0) * s1 + d1 mod s1) mod 8192), meshShape = , chipIds = [0]>
#system_desc = #tt.system_desc<[{role = host, target_triple = "x86_64-pc-linux"}], [{arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99104, erisc_l1_unreserved_base = 104480, dram_unreserved_base = 32, dram_unreserved_end = 1073196736, physical_cores = {worker = [ 18x18,  18x19,  18x20,  18x21,  18x22,  18x23,  18x24,  18x25,  19x18,  19x19,  19x20,  19x21,  19x22,  19x23,  19x24,  19x25,  20x18,  20x19,  20x20,  20x21,  20x22,  20x23,  20x24,  20x25,  21x18,  21x19,  21x20,  21x21,  21x22,  21x23,  21x24,  21x25,  22x18,  22x19,  22x20,  22x21,  22x22,  22x23,  22x24,  22x25,  23x18,  23x19,  23x20,  23x21,  23x22,  23x23,  23x24,  23x25,  24x18,  24x19,  24x20,  24x21,  24x22,  24x23,  24x24,  24x25,  25x18,  25x19,  25x20,  25x21,  25x22,  25x23,  25x24,  25x25] dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth = [ 17x25] eth_inactive = [ 16x18,  16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  16x25,  17x19,  17x20,  17x22,  17x23,  17x24]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}, {arch = <wormhole_b0>, grid = 8x8, l1_size = 1499136, num_dram_channels = 12, dram_channel_size = 1073741824, noc_l1_address_align_bytes = 16, pcie_address_align_bytes = 32, noc_dram_address_align_bytes = 32, l1_unreserved_base = 99104, erisc_l1_unreserved_base = 104480, dram_unreserved_base = 32, dram_unreserved_end = 1073196736, physical_cores = {worker = [ 18x18,  18x19,  18x20,  18x21,  18x22,  18x23,  18x24,  18x25,  19x18,  19x19,  19x20,  19x21,  19x22,  19x23,  19x24,  19x25,  20x18,  20x19,  20x20,  20x21,  20x22,  20x23,  20x24,  20x25,  21x18,  21x19,  21x20,  21x21,  21x22,  21x23,  21x24,  21x25,  22x18,  22x19,  22x20,  22x21,  22x22,  22x23,  22x24,  22x25,  23x18,  23x19,  23x20,  23x21,  23x22,  23x23,  23x24,  23x25,  24x18,  24x19,  24x20,  24x21,  24x22,  24x23,  24x24,  24x25,  25x18,  25x19,  25x20,  25x21,  25x22,  25x23,  25x24,  25x25] dram = [ 0x0,  0x1,  0x2,  0x3,  0x4,  0x5,  0x6,  0x7,  0x8,  0x9,  0x10,  0x11] eth = [ 16x25] eth_inactive = [ 16x19,  16x20,  16x21,  16x22,  16x23,  16x24,  17x18,  17x19,  17x20,  17x21,  17x22,  17x23,  17x24,  17x25]}, supported_data_types = [<f32>, <f16>, <bf16>, <bfp_f8>, <bfp_bf8>, <bfp_f4>, <bfp_bf4>, <bfp_f2>, <bfp_bf2>, <u32>, <u16>, <u8>], supported_tile_sizes = [ 4x16,  16x16,  32x16,  4x32,  16x32,  32x32], num_cbs = 32}], [0, 1], [3 : i32, 0 : i32], [ 0x0x0x0], [<[0, 8, 0], [1, 0, 0]>]>
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_host_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
#ttnn_layout_host_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #system_memory>>
#ttnn_layout_host_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, f32>, #system_memory>>
#ttnn_layout_host_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, bf16>, #system_memory>>
#ttnn_layout_device_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #dram>, <interleaved>>
#ttnn_layout_device_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_device_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_device_tile_u32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout_device_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {tt.device = #device, tt.system_desc = #system_desc} {
    // Test cases when we do layout transformation from host and we don't change tensor layout and tensor data type
    //

    // Test case when we move tensor from host to device.
    func.func @from_host_to_device_layout_to_layout_dt_to_dt_create_to_device_op(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_device_rm> {
        // Verify that we only insert the to_device op when there are no layout or data type changes.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%arg0, %[[GET_DEVICE_OP]])
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <<64x128>>, <interleaved>>
        // CHECK: return %[[TO_DEVICE_OP]]
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
        %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <<64x128>>, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>, !tt.device<#device>) -> tensor<64x128xf32, #ttnn_layout_device_rm>
        return %1 : tensor<64x128xf32, #ttnn_layout_device_rm>
    }

    // Test cases when we do layout transformation from host and we don't change tensor layout but we cast tensor data type.
    //

    // Test case when we move tensor from host to host for tile case.
    func.func @from_host_to_host_layout_to_layout_create_data_cast_op_tile(%arg0: tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xbf16, #ttnn_layout_host_tile_bf16> {
        // Typecast works only on device. Verify that for the tile case when the output is on host, we insert the to_dtype op to cast the data type on host.
        // CHECK: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
        // CHECK-NEXT: return %[[CASTING_OP]]
        %1 = "ttnn.to_layout"(%arg0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#system_memory, <<2x4>>>}> : (tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>
        return %1 : tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>
    }

    // Test case when we move tensor from host to host for row-major case.
    func.func @from_host_to_host_layout_to_layout_create_data_cast_op_rm(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xbf16, #ttnn_layout_host_rm_bf16> {
        // Typecast works only on device. Verify that for the row-major case when the output is on host, we insert the to_dtype op to cast the data type on host.
        // CHECK: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
        // CHECK-NEXT: return %[[CASTING_OP]]
        %1 = "ttnn.to_layout"(%arg0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory, <<64x128>>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>
        return %1 : tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>
    }

    // Test case when we move tensor from host to device for row-major case.
    func.func @from_host_to_device_layout_to_layout_create_data_cast_op_rm(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xbf16, #ttnn_layout_device_rm_bf16> {
        // Typecast on device only works for tile layout. Verify that for the row-major case we insert the to_dtype op to cast the data type on host and than move the tensor to device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
        // CHECK: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%[[CASTING_OP]], %[[GET_DEVICE_OP]])
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <<64x128>>, <interleaved>>
        // CHECK-NEXT: return %[[TO_DEVICE_OP]]
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
        %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <<64x128>>, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>, !tt.device<#device>) -> tensor<64x128xbf16, #ttnn_layout_device_rm_bf16>
        return %1 : tensor<64x128xbf16, #ttnn_layout_device_rm_bf16>
    }

    // Test case when we move tensor from host to device for tile case.
    func.func @from_host_to_device_layout_to_layout_create_data_cast_op_tile(%arg0: tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16> {
        // Typecast on device only works for tile layout. Verify that for the tile case we insert the to_device op and the typecast op to cast the data type on device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%arg0, %[[GET_DEVICE_OP]])
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <<2x4>>, <interleaved>>
        // CHECK-NEXT: %[[CASTING_OP:.*]] = "ttnn.typecast"(%[[TO_DEVICE_OP]])
        // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
        // CHECK-NEXT: return %[[CASTING_OP]]
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
        %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<2x4>>, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_tile>, !tt.device<#device>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
        return %1 : tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
    }

    // Test cases when we do layout transformation from host and we change tensor layout but we don't cast tensor data type.
    //

    // Test case when we move tensor from host to host for tile -> row-major case.
    func.func @from_host_to_host_dt_to_dt_from_tile_to_rm(%arg0: tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xf32, #ttnn_layout_host_rm> {
        // This test verifies that the `to_layout` operation is correctly inserted to change the layout from tile to row-major on the host.
        // CHECK: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK-NEXT: return %[[TO_LAYOUT_OP]]
        %1 = "ttnn.to_layout"(%arg0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory, <<64x128>>>}> : (tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xf32, #ttnn_layout_host_rm>
        return %1 : tensor<64x128xf32, #ttnn_layout_host_rm>
    }

    // Test case when we move tensor from host to host for row-major -> tile case.
    func.func @from_host_to_host_dt_to_dt_from_rm_to_tile(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_host_tile> {
        // This test verifies that the `to_layout` operation is correctly inserted to change the layout from row-major to tile on the host.
        // CHECK: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: return %[[TO_LAYOUT_OP]]
        %1 = "ttnn.to_layout"(%arg0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#system_memory, <<2x4>>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_host_tile>
        return %1 : tensor<64x128xf32, #ttnn_layout_host_tile>
    }

    // Test case when we move tensor from host to device for tile -> row-major case.
    func.func @from_host_to_device_dt_to_dt_from_tile_to_rm(%arg0: tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xf32, #ttnn_layout_device_rm> {
        // This test verifies that the `to_layout` and `to_device` operations are correctly inserted to change the layout from tile to row-major on the host and than move the tensor to the device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%[[TO_LAYOUT_OP]], %[[GET_DEVICE_OP]])
        // CHECK-NEXT: return %[[TO_DEVICE_OP]]
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
        %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <<64x128>>, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_tile>, !tt.device<#device>) -> tensor<64x128xf32, #ttnn_layout_device_rm>
        return %1 : tensor<64x128xf32, #ttnn_layout_device_rm>
    }

    // Test case when we move tensor from host to device for row-major -> tile case for bf16 data type.
    func.func @from_host_to_device_dt_to_dt_from_rm_to_tile_bf16(%arg0: tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16> {
        // This test verifies that the `to_device` and `to_layout` operations are correctly inserted to change the layout from row-major to tile on the device.
        // Specifically, it ensures that BF16 tiling is performed on the device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%arg0, %[[GET_DEVICE_OP]])
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[TO_DEVICE_OP]], %[[GET_DEVICE_OP]])
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: return %[[TO_LAYOUT_OP]]
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
        %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<2x4>>, <interleaved>>}> : (tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>, !tt.device<#device>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
        return %1 : tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
    }

    // Test case when we move tensor from host to device for row-major -> tile case for non-bf16 data type.
    func.func @from_host_to_device_dt_to_dt_from_rm_to_tile_f32(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_device_tile> {
        // This test verifies that the `to_layout` and `to_device` operations are correctly inserted to change the layout from row-major to tile on the host for non bf16 data type.
        // Specifically, it ensures that non-BF16 tiling is performed on the host and then moved to the device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%[[TO_LAYOUT_OP]], %[[GET_DEVICE_OP]])
        // CHECK-NEXT: return %[[TO_DEVICE_OP]]
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
        %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<2x4>>, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>, !tt.device<#device>) -> tensor<64x128xf32, #ttnn_layout_device_tile>
        return %1 : tensor<64x128xf32, #ttnn_layout_device_tile>
    }

    // Test cases when we do layout transformation from host and we change both tensor layout and tensor data type.
    //

    // Test case when we move tensor from host to host for tile -> row-major case and data type cast.
    func.func @from_host_to_host_from_bf16_to_f32_from_tile_to_rm(%arg0: tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>) -> tensor<64x128xf32, #ttnn_layout_host_rm> {
        // This test verifies that the `to_layout` and `to_dtype` operations are correctly inserted to change the layout from tile to row-major and cast data type from bf16 to f32 on host.
        // CHECK: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #tt.supportedDataTypes<f32>
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[CASTING_OP]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK-NEXT: return %[[TO_LAYOUT_OP]]
        %1 = "ttnn.to_layout"(%arg0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory, <<64x128>>>}> : (tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>) -> tensor<64x128xf32, #ttnn_layout_host_rm>
        return %1 : tensor<64x128xf32, #ttnn_layout_host_rm>
    }

    // Test case when we move tensor from host to host for row-major -> tile case and data type cast.
    func.func @from_host_to_host_from_bf16_to_f32_from_rm_to_tile(%arg0: tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>) -> tensor<64x128xf32, #ttnn_layout_host_tile> {
        // This test verifies that the `to_layout` and `to_dtype` operations are correctly inserted to change the layout from row-major to tile and cast data type from bf16 to f32 on host.
        // CHECK: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #tt.supportedDataTypes<f32>
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[CASTING_OP]])
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: return %[[TO_LAYOUT_OP]]
        %1 = "ttnn.to_layout"(%arg0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#system_memory, <<2x4>>>}> : (tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>) -> tensor<64x128xf32, #ttnn_layout_host_tile>
        return %1 : tensor<64x128xf32, #ttnn_layout_host_tile>
    }

    // Test case when we move tensor from host to device for tile -> row-major case and cast input from bf16.
    func.func @from_host_to_device_data_type_from_bf16_to_f32_from_tile_to_rm(%arg0: tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>) -> tensor<64x128xf32, #ttnn_layout_device_rm> {
        // This test verifies that the `to_dtype`, `to_layout` and `to_device` operations are correctly inserted to change the layout from tile to row-major and cast data type from bf16 to f32 on host and then move tensor to device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #tt.supportedDataTypes<f32>
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[CASTING_OP]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%[[TO_LAYOUT_OP]], %[[GET_DEVICE_OP]])
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <<64x128>>, <interleaved>>
        // CHECK-NEXT: return %[[TO_DEVICE_OP]]
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
        %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <<64x128>>, <interleaved>>}> : (tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>, !tt.device<#device>) -> tensor<64x128xf32, #ttnn_layout_device_rm>
        return %1 : tensor<64x128xf32, #ttnn_layout_device_rm>
    }

    // Test case when we move tensor from host to device for row-major -> tile case and cast input from bf16.
    func.func @from_host_to_device_data_type_from_bf16_to_f32_from_rm_to_tile(%arg0: tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>) -> tensor<64x128xf32, #ttnn_layout_device_tile> {
        // This test verifies that the `to_device`, `to_layout` and `typecast` operations are correctly inserted to change the layout from row-major to tile and cast
        // data type from bf16 to f32 on device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%arg0, %[[GET_DEVICE_OP]])
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[TO_DEVICE_OP]], %[[GET_DEVICE_OP]])
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: %[[CASTING_OP:.*]] = "ttnn.typecast"(%[[TO_LAYOUT_OP]])
        // CHECK-SAME: dtype = #tt.supportedDataTypes<f32>
        // CHECK-NEXT: return %[[CASTING_OP]]
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
        %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<2x4>>, <interleaved>>}> : (tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>, !tt.device<#device>) -> tensor<64x128xf32, #ttnn_layout_device_tile>
        return %1 : tensor<64x128xf32, #ttnn_layout_device_tile>
    }

    // Test case when we move tensor from host to device for row-major -> tile case and cast input to bf16.
    func.func @from_host_to_device_data_type_from_f32_to_bf16_from_rm_to_tile(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16> {
        // This test verifies that the `to_dtype`, `to_device` and `to_layout` operations are correctly inserted to cast the data type from f32 to bf16 on host and then move tensor to device and change the layout from row-major to tile.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #tt.supportedDataTypes<bf16>
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%[[CASTING_OP]], %[[GET_DEVICE_OP]])
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[TO_DEVICE_OP]], %[[GET_DEVICE_OP]])
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: return %[[TO_LAYOUT_OP]]
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
        %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<2x4>>, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>, !tt.device<#device>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
        return %1 : tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
    }

    // Test case when we move tensor from host to device for row-major -> tile case and we don't cast data type to bf16 nor from bf16.
    func.func @from_host_to_device_data_type_from_f32_to_u32_from_rm_to_tile(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xui32, #ttnn_layout_device_tile_u32> {
        // This test verifies that the `to_dtype`, `to_layout` and `to_device` operations are correctly inserted to cast the data type from f32 to f16 and tilize on host and then move tensor to device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #tt.supportedDataTypes<u32>
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[CASTING_OP]])
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%[[TO_LAYOUT_OP]], %[[GET_DEVICE_OP]])
        // CHECK-NEXT: return %[[TO_DEVICE_OP]]
        %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !tt.device<#device>
        %1 = "ttnn.to_layout"(%arg0, %0) <{dtype = #tt.supportedDataTypes<u32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <<2x4>>, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>, !tt.device<#device>) -> tensor<64x128xui32, #ttnn_layout_device_tile_u32>
        return %1 : tensor<64x128xui32, #ttnn_layout_device_tile_u32>
    }
}
