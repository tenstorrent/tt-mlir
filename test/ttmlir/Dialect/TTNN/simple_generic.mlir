#any_device = #tt.operand_constraint<dram|l1|scalar|tile|any_device|any_device_tile>
module attributes {} {
  func.func @forward(%arg0: tensor<64x128xf32>, %arg1: tensor<64x128xf32>) -> tensor<64x128xf32> {
    // CHECK: %[[C:.*]] = "ttnn.open_device"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.full"[[C:.*]]
    %0 = tensor.empty() : tensor<64x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.to_memory_config"[[C:.*]]
    // CHECK: %[[C:.*]] = "ttnn.generic"[[C:.*]]
    %1 = "ttir.external_generic"(%arg0, %arg1, %0) <{
      circular_buffer_attributes = [
        #tt.circular_buffer_attributes<c_in0, <0x0, 6x6>, 4096, 2048, bf16>, 
        #tt.circular_buffer_attributes<c_in1, <0x0, 6x6>, 4096, 2048, bf16>, 
        #tt.circular_buffer_attributes<c_out0, <0x0, 6x6>, 4096, 2048, bf16>
      ], 
      data_movement_attributes = [
        #tt.data_movement_attributes<
          <0x0, 6x6>, 
          "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/dataflow/reader_binary_interleaved_start_id.cpp", 
          <reader, <1, 1>>
        >, 
        #tt.data_movement_attributes<
          <0x0, 6x6>, 
          "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp", 
          <writer, <1, 1>>>
      ], 
      compute_attributes = [
        #tt.compute_attributes<
          <0x0, 6x6>, 
          "ttnn/cpp/ttnn/operations/eltwise/binary/device/kernels/compute/eltwise_binary_kernel.cpp", 
          <hifi4, false, false, false, <1>, {ELTWISE_OP = "add_tiles", ELTWISE_OP_TYPE = "EltwiseBinaryType::ELWADD"}>
        >
      ], 
      operandSegmentSizes = array<i32: 2, 1>, 
      operand_constraints = [#any_device, #any_device, #any_device]
    }> : (tensor<64x128xf32>, tensor<64x128xf32>, tensor<64x128xf32>) -> tensor<64x128xf32>
    // CHECK: %[[C:.*]] = "ttnn.to_memory_config"[[C:.*]]
    // CHECK: "ttnn.close_device"[[C:.*]]
    return %1 : tensor<64x128xf32>
  }
}
