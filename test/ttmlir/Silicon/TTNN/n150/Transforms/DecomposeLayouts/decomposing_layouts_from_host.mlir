// RUN: python %ttmlir_scripts_root/extract-and-replace-system-desc-and-device.py --input-file %s --temp-dir %T --system-desc-path "%system_desc_path%"  > %t.mlir
// RUN: ttmlir-opt --ttnn-decompose-layouts %t.mlir > %t_ttnn_mlir.mlir
// RUN: FileCheck %t.mlir --input-file=%t_ttnn_mlir.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t_ttnn_mlir.mlir > %t.ttnn
#device = #tt.device<>
#system_desc = #tt.system_desc<>
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
