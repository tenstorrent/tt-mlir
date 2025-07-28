// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-decompose-layouts -o %t %s
// RUN: FileCheck %s --input-file=%t
#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_host_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
#ttnn_layout_host_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #system_memory>>
#ttnn_layout_host_rm_u32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xui32, #system_memory>>
#ttnn_layout_host_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #system_memory>>
#ttnn_layout_host_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #system_memory>>
#ttnn_layout_device_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #dram>, <interleaved>>
#ttnn_layout_device_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_device_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout_device_tile_u32 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, u32>, #dram>, <interleaved>>
#ttnn_layout_device_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #dram>, <interleaved>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
module attributes {} {

    // Test cases when we do layout transformation from host and we don't change tensor layout and tensor data type
    //

    // Test case when we move tensor from host to device.
    func.func @from_host_to_device_layout_to_layout_dt_to_dt_create_to_device_op(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_device_rm> {
        // Verify that we only insert the to_device op when there are no layout or data type changes.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%arg0, %[[GET_DEVICE_OP]])
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
        // CHECK: return %[[TO_DEVICE_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_device_rm>
        return %0 : tensor<64x128xf32, #ttnn_layout_device_rm>
    }

    // Test cases when we do layout transformation from host and we don't change tensor layout but we cast tensor data type.
    //

    // Test case when we move tensor from host to host for tile case.
    func.func @from_host_to_host_layout_to_layout_create_data_cast_op_tile(%arg0: tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xbf16, #ttnn_layout_host_tile_bf16> {
        // Typecast works only on device. Verify that for the tile case when the output is on host, we insert the to_dtype op to cast the data type on host.
        // CHECK: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
        // CHECK-NEXT: return %[[CASTING_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>
        return %0 : tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>
    }

    // Test case when we move tensor from host to host for row-major case.
    func.func @from_host_to_host_layout_to_layout_create_data_cast_op_rm(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xbf16, #ttnn_layout_host_rm_bf16> {
        // Typecast works only on device. Verify that for the row-major case when the output is on host, we insert the to_dtype op to cast the data type on host.
        // CHECK: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
        // CHECK-NEXT: return %[[CASTING_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>
        return %0 : tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>
    }

    // Test case when we move tensor from host to device for row-major case.
    func.func @from_host_to_device_layout_to_layout_create_data_cast_op_rm(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xbf16, #ttnn_layout_device_rm_bf16> {
        // Typecast on device only works for tile layout. Verify that for the row-major case we insert the to_dtype op to cast the data type on host and than move the tensor to device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
        // CHECK: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%[[CASTING_OP]], %[[GET_DEVICE_OP]])
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
        // CHECK-NEXT: return %[[TO_DEVICE_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xbf16, #ttnn_layout_device_rm_bf16>
        return %0 : tensor<64x128xbf16, #ttnn_layout_device_rm_bf16>
    }

    // Test case when we move tensor from host to device for tile case.
    func.func @from_host_to_device_layout_to_layout_create_data_cast_op_tile(%arg0: tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16> {
        // Typecast on device only works for tile layout. Verify that for the tile case we insert the to_device op and the typecast op to cast the data type on device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%arg0, %[[GET_DEVICE_OP]])
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
        // CHECK-NEXT: %[[CASTING_OP:.*]] = "ttnn.typecast"(%[[TO_DEVICE_OP]])
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
        // CHECK-NEXT: return %[[CASTING_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
        return %0 : tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
    }

    // Test cases when we do layout transformation from host and we change tensor layout but we don't cast tensor data type.
    //

    // Test case when we move tensor from host to host for tile -> row-major case.
    func.func @from_host_to_host_dt_to_dt_from_tile_to_rm(%arg0: tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xf32, #ttnn_layout_host_rm> {
        // This test verifies that the `to_layout` operation is correctly inserted to change the layout from tile to row-major on the host.
        // CHECK: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK-NEXT: return %[[TO_LAYOUT_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xf32, #ttnn_layout_host_rm>
        return %0 : tensor<64x128xf32, #ttnn_layout_host_rm>
    }

    // Test case when we move tensor from host to host for row-major -> tile case.
    func.func @from_host_to_host_dt_to_dt_from_rm_to_tile(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_host_tile> {
        // This test verifies that the `to_layout` operation is correctly inserted to change the layout from row-major to tile on the host.
        // CHECK: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: return %[[TO_LAYOUT_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_host_tile>
        return %0 : tensor<64x128xf32, #ttnn_layout_host_tile>
    }

    // Test case when we move tensor from host to device for tile -> row-major case for bf16 data type.
    func.func @from_host_to_device_dt_to_dt_from_tile_to_rm_bf16(%arg0: tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>) -> tensor<64x128xbf16, #ttnn_layout_device_rm_bf16> {
        // This test verifies that the `to_layout` and `to_device` operations are correctly inserted to move the tensor to the device and then change layout from tile to row-major.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%arg0, %[[GET_DEVICE_OP]])
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[TO_DEVICE_OP]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK-NEXT: return %[[TO_LAYOUT_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>) -> tensor<64x128xbf16, #ttnn_layout_device_rm_bf16>
        return %0 : tensor<64x128xbf16, #ttnn_layout_device_rm_bf16>
    }

    // Test case when we move tensor from host to device for tile -> row-major case for f32 data type.
    func.func @from_host_to_device_dt_to_dt_from_tile_to_rm_f32(%arg0: tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xf32, #ttnn_layout_device_rm> {
        // This test verifies that the `to_layout` and `to_device` operations are correctly inserted to move the tensor to the device and then change layout from tile to row-major.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%[[TO_LAYOUT_OP]], %[[GET_DEVICE_OP]])
        // CHECK-NEXT: return %[[TO_DEVICE_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_tile>) -> tensor<64x128xf32, #ttnn_layout_device_rm>
        return %0 : tensor<64x128xf32, #ttnn_layout_device_rm>
    }

    // Test case when we move tensor from host to device for row-major -> tile case for bf16 data type.
    func.func @from_host_to_device_dt_to_dt_from_rm_to_tile_bf16(%arg0: tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16> {
        // This test verifies that the `to_device` and `to_layout` operations are correctly inserted to change the layout from row-major to tile on the device.
        // Specifically, it ensures that BF16 tiling is performed on the device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%arg0, %[[GET_DEVICE_OP]])
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[TO_DEVICE_OP]])
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: return %[[TO_LAYOUT_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
        return %0 : tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
    }

    // Test case when we move tensor from host to device for row-major -> tile case for non-bf16 non-f32 data type.
    func.func @from_host_to_device_dt_to_dt_from_rm_to_tile_f16(%arg0: tensor<64x128xui32, #ttnn_layout_host_rm_u32>) -> tensor<64x128xui32, #ttnn_layout_device_tile_u32> {
        // This test verifies that the `to_layout` and `to_device` operations are correctly inserted to change the layout from row-major to tile on the host for non bf16 non f32 data type.
        // Specifically, it ensures that non-BF16 non-FP32 tiling is performed on the host and then moved to the device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%arg0)
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%[[TO_LAYOUT_OP]], %[[GET_DEVICE_OP]])
        // CHECK-NEXT: return %[[TO_DEVICE_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<u32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xui32, #ttnn_layout_host_rm_u32>) -> tensor<64x128xui32, #ttnn_layout_device_tile_u32>
        return %0 : tensor<64x128xui32, #ttnn_layout_device_tile_u32>
    }

    // Test cases when we do layout transformation from host and we change both tensor layout and tensor data type.
    //

    // Test case when we move tensor from host to host for tile -> row-major case and data type cast.
    func.func @from_host_to_host_from_bf16_to_f32_from_tile_to_rm(%arg0: tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>) -> tensor<64x128xf32, #ttnn_layout_host_rm> {
        // This test verifies that the `to_layout` and `to_dtype` operations are correctly inserted to change the layout from tile to row-major and cast data type from bf16 to f32 on host.
        // CHECK: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[CASTING_OP]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK-NEXT: return %[[TO_LAYOUT_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>) -> tensor<64x128xf32, #ttnn_layout_host_rm>
        return %0 : tensor<64x128xf32, #ttnn_layout_host_rm>
    }

    // Test case when we move tensor from host to host for row-major -> tile case and data type cast.
    func.func @from_host_to_host_from_bf16_to_f32_from_rm_to_tile(%arg0: tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>) -> tensor<64x128xf32, #ttnn_layout_host_tile> {
        // This test verifies that the `to_layout` and `to_dtype` operations are correctly inserted to change the layout from row-major to tile and cast data type from bf16 to f32 on host.
        // CHECK: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[CASTING_OP]])
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: return %[[TO_LAYOUT_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>) -> tensor<64x128xf32, #ttnn_layout_host_tile>
        return %0 : tensor<64x128xf32, #ttnn_layout_host_tile>
    }

    // Test case when we move tensor from host to device for tile -> row-major case and cast input from bf16.
    func.func @from_host_to_device_data_type_from_bf16_to_f32_from_tile_to_rm(%arg0: tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>) -> tensor<64x128xf32, #ttnn_layout_device_rm> {
        // This test verifies that the `to_dtype`, `to_layout` and `to_device` operations are correctly inserted to change the layout from tile to row-major and cast data type from bf16 to f32 on host and then move tensor to device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[CASTING_OP]])
        // CHECK-SAME: layout = #ttnn.layout<row_major>
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%[[TO_LAYOUT_OP]], %[[GET_DEVICE_OP]])
        // CHECK-SAME: memory_config = #ttnn.memory_config<#dram, <interleaved>>
        // CHECK-NEXT: return %[[TO_DEVICE_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xbf16, #ttnn_layout_host_tile_bf16>) -> tensor<64x128xf32, #ttnn_layout_device_rm>
        return %0 : tensor<64x128xf32, #ttnn_layout_device_rm>
    }

    // Test case when we move tensor from host to device for row-major -> tile case and cast input from bf16.
    func.func @from_host_to_device_data_type_from_bf16_to_f32_from_rm_to_tile(%arg0: tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>) -> tensor<64x128xf32, #ttnn_layout_device_tile> {
        // This test verifies that the `to_device`, `to_layout` and `typecast` operations are correctly inserted to change the layout from row-major to tile and cast
        // data type from bf16 to f32 on device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%arg0, %[[GET_DEVICE_OP]])
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[TO_DEVICE_OP]])
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: %[[CASTING_OP:.*]] = "ttnn.typecast"(%[[TO_LAYOUT_OP]])
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
        // CHECK-NEXT: return %[[CASTING_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xbf16, #ttnn_layout_host_rm_bf16>) -> tensor<64x128xf32, #ttnn_layout_device_tile>
        return %0 : tensor<64x128xf32, #ttnn_layout_device_tile>
    }

    // Test case when we move tensor from host to device for row-major -> tile case and cast input to bf16.
    func.func @from_host_to_device_data_type_from_f32_to_bf16_from_rm_to_tile(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16> {
        // This test verifies that the `to_dtype`, `to_device` and `to_layout` operations are correctly inserted to cast the data type from f32 to bf16 on host and then move tensor to device and change the layout from row-major to tile.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%[[CASTING_OP]], %[[GET_DEVICE_OP]])
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[TO_DEVICE_OP]])
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: return %[[TO_LAYOUT_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
        return %0 : tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
    }

    // Test case when we move tensor from host to device for row-major -> tile case and we don't cast data type to bf16 nor from bf16.
    func.func @from_host_to_device_data_type_from_f32_to_u32_from_rm_to_tile(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xui32, #ttnn_layout_device_tile_u32> {
        // This test verifies that the `to_dtype`, `to_layout` and `to_device` operations are correctly inserted to cast the data type from f32 to f16 and tilize on host and then move tensor to device.
        // CHECK: %[[GET_DEVICE_OP:.*]] = "ttnn.get_device"()
        // CHECK-NEXT: %[[CASTING_OP:.*]] = "ttnn.to_dtype"(%arg0)
        // CHECK-SAME: dtype = #ttcore.supportedDataTypes<u32>
        // CHECK-NEXT: %[[TO_LAYOUT_OP:.*]] = "ttnn.to_layout"(%[[CASTING_OP]])
        // CHECK-SAME: layout = #ttnn.layout<tile>
        // CHECK-NEXT: %[[TO_DEVICE_OP:.*]] = "ttnn.to_device"(%[[TO_LAYOUT_OP]], %[[GET_DEVICE_OP]])
        // CHECK-NEXT: return %[[TO_DEVICE_OP]]
        %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<u32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xui32, #ttnn_layout_device_tile_u32>
        return %0 : tensor<64x128xui32, #ttnn_layout_device_tile_u32>
    }
}
