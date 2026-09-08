// REQUIRES: opmodel
// RUN: ttmlir-opt --ttir-to-ttnn-backend-pipeline="optimization-level=2" -o %t %s
// RUN: FileCheck %s --input-file=%t

module {
  // The TTML kernel requires tiled, DRAM-interleaved operands and derives both
  // output layouts from the input.
  func.func @rmsnorm_fw(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>) {
    // CHECK-DAG: #[[INPUT_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[GAMMA_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<1x8x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>
    // CHECK-DAG: #[[RMS_LAYOUT:ttnn_layout[0-9]*]] = #ttnn.ttnn_layout<{{.*}}memref<4x1x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

    // CHECK-LABEL: func.func @rmsnorm_fw(
    // CHECK-SAME: %[[INPUT:[0-9a-z_]+]]: tensor<1x1x128x256xbf16, #[[INPUT_LAYOUT]]>
    // CHECK-SAME: %[[GAMMA:[0-9a-z_]+]]: tensor<1x1x1x256xbf16, #[[GAMMA_LAYOUT]]>)
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16, #[[INPUT_LAYOUT]]>, tensor<1x1x128x1xbf16, #[[RMS_LAYOUT]]>)
    // CHECK: %[[OUTPUT:[0-9a-z_]+]], %[[RMS:[0-9a-z_]+]] = "ttnn.rmsnorm_fw"(%[[INPUT]], %[[GAMMA]])
    // CHECK-SAME: -> (tensor<1x1x128x256xbf16, #[[INPUT_LAYOUT]]>, tensor<1x1x128x1xbf16, #[[RMS_LAYOUT]]>)
    // CHECK: return %[[OUTPUT]], %[[RMS]]
    %output, %rms = "ttcore.composite"(%input, %gamma) <{
        composite_name = "rmsnorm_fw",
        decomposition = @rmsnorm_fw_intermediates_decomp,
        composite_attributes = {
          return_intermediates = true,
          epsilon = 1.000000e-06 : f32}}>
        : (tensor<1x1x128x256xbf16>, tensor<1x1x1x256xbf16>)
          -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>)
    return %output, %rms
        : tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>
  }

  // Exercise the bf16 workaround together with OpModel validation while
  // preserving the f32 function boundary.
  func.func @rmsnorm_fw_f32(
      %input: tensor<1x1x128x256xf32>,
      %gamma: tensor<1x1x1x256xf32>) -> tensor<1x1x128x256xf32> {
    // CHECK-LABEL: func.func @rmsnorm_fw_f32(
    // CHECK: %[[INPUT_BF16:[0-9a-z_]+]] = "ttnn.typecast"(%{{.*}})
    // CHECK-SAME: -> tensor<1x1x128x256xbf16, #[[INPUT_LAYOUT]]>
    // CHECK: %[[GAMMA_BF16:[0-9a-z_]+]] = "ttnn.typecast"(%{{.*}})
    // CHECK-SAME: -> tensor<1x1x1x256xbf16, #[[GAMMA_LAYOUT]]>
    // CHECK: %[[OUTPUT_BF16:[0-9a-z_]+]] = "ttnn.rmsnorm_fw"(%[[INPUT_BF16]], %[[GAMMA_BF16]])
    // CHECK-SAME: -> tensor<1x1x128x256xbf16, #[[INPUT_LAYOUT]]>
    // CHECK: %[[OUTPUT_F32:[0-9a-z_]+]] = "ttnn.typecast"(%[[OUTPUT_BF16]])
    // CHECK: return %[[OUTPUT_F32]]
    %output = "ttcore.composite"(%input, %gamma) <{
        composite_name = "rmsnorm_fw",
        decomposition = @rmsnorm_fw_decomp,
        composite_attributes = {
          return_intermediates = false,
          epsilon = 1.000000e-06 : f32}}>
        : (tensor<1x1x128x256xf32>, tensor<1x1x1x256xf32>)
          -> tensor<1x1x128x256xf32>
    return %output : tensor<1x1x128x256xf32>
  }

  func.func private @rmsnorm_fw_decomp(
      %input: tensor<1x1x128x256xf32>,
      %gamma: tensor<1x1x1x256xf32>) -> tensor<1x1x128x256xf32> {
    return %input : tensor<1x1x128x256xf32>
  }

  func.func private @rmsnorm_fw_intermediates_decomp(
      %input: tensor<1x1x128x256xbf16>,
      %gamma: tensor<1x1x1x256xbf16>)
      -> (tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>) {
    %rms = "ttir.zeros"() <{shape = array<i32: 1, 1, 128, 1>}>
        : () -> tensor<1x1x128x1xbf16>
    return %input, %rms
        : tensor<1x1x128x256xbf16>, tensor<1x1x128x1xbf16>
  }
}
