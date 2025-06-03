// RUN: ttmlir-opt --tt-register-device="system-desc-path=%system_desc_path%" %s > %t.mlir
// RUN: ttmlir-translate --ttnn-to-flatbuffer %t.mlir > %t.ttnn

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 1024 + d1 * 32 + d2, d3), <1x1>, memref<1024x64xbf16, #system_memory>>
#ttnn_layout1 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 192 + d1 * 3 + d2, d3), <1x1>, memref<12288x3xbf16, #system_memory>>
#ttnn_layout2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 + d1 + d2, d3), <1x1>, memref<1x64xbf16, #system_memory>>
#ttnn_layout3 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 900 + d1 * 30 + d2, d3), <1x1>, memref<900x64xbf16, #system_memory>>
#ttnn_layout4 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 900 + d1 * 30 + d2, d3), <1x1>, memref<29x2x!tt.tile<32x32, bf16>, #dram>, <interleaved>>
#ttnn_layout5 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 900 + d1 * 30 + d2, d3), <1x1>, memref<29x2x!tt.tile<32x32, bf16>, #system_memory>>

#device_compute_kernel_config = #ttnn.device_compute_kernel_config<
  math_fidelity = lofi,
  math_approx_mode = true,
  fp32_dest_acc_en = false,
  packer_l1_acc = false,
  dst_full_sync_en = false
>

module attributes {} {
  func.func @forward(%arg0: tensor<1x1x1024x64xbf16, #ttnn_layout>, %arg1: tensor<64x64x3x3xbf16, #ttnn_layout1>, %arg2: tensor<1x1x1x64xbf16, #ttnn_layout2>) -> tensor<1x30x30x64xbf16, #ttnn_layout3> {
    %0 = "ttnn.get_device"() <{mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %1 = "ttnn.conv2d"(%arg0, %arg1, %arg2, %0)
            <{
              in_channels = 64 : i32,
              out_channels = 64 : i32,
              batch_size = 1 : i32,
              input_height = 32 : i32,
              input_width = 32 : i32,
              kernel_size = array<i32: 3, 3>,
              stride = array<i32: 1, 1>,
              padding = array<i32: 0, 0>,
              dilation = array<i32: 1, 1>,
              groups = 1 : i32,
              compute_config = #device_compute_kernel_config
            }> : (tensor<1x1x1024x64xbf16, #ttnn_layout>, tensor<64x64x3x3xbf16, #ttnn_layout1>, tensor<1x1x1x64xbf16, #ttnn_layout2>, !ttnn.device) -> tensor<1x1x900x64xbf16, #ttnn_layout4>
    %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 30 : i32, 30 : i32, 64 : i32]}> : (tensor<1x1x900x64xbf16, #ttnn_layout4>) -> tensor<1x30x30x64xbf16, #ttnn_layout4>
    "ttnn.deallocate"(%1) <{force = false}> : (tensor<1x1x900x64xbf16, #ttnn_layout4>) -> ()
    %3 = "ttnn.from_device"(%2) : (tensor<1x30x30x64xbf16, #ttnn_layout4>) -> tensor<1x30x30x64xbf16, #ttnn_layout5>
    "ttnn.deallocate"(%2) <{force = false}> : (tensor<1x30x30x64xbf16, #ttnn_layout4>) -> ()
    %4 = "ttnn.to_layout"(%3) <{layout = #ttnn.layout<row_major>}> : (tensor<1x30x30x64xbf16, #ttnn_layout5>) -> tensor<1x30x30x64xbf16, #ttnn_layout3>
    "ttnn.deallocate"(%3) <{force = false}> : (tensor<1x30x30x64xbf16, #ttnn_layout5>) -> ()
    return %4 : tensor<1x30x30x64xbf16, #ttnn_layout3>
  }
}
