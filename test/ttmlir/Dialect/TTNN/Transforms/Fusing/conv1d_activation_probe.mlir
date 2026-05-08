// RUN: ttmlir-opt --ttnn-fusing -o %t %s
// RUN: FileCheck %s --input-file=%t

// Pilot 3.2 probe: verify TTNNConv2dWithActivation peels through ONE
// post-conv `ttnn.reshape` between conv2d and the activation. This is the
// pattern produced by conv1d lowering (the trailing 4D->3D reshape that
// recovers the conv1d output rank), and by Whisper-style conv1d+gelu blocks.
//
// At the TTNN level the canonical chain after lowering a conv1d->activation
// is `ttnn.conv2d -> ttnn.relu` (the trailing reshape/permute back to rank-3
// is sunk *after* the activation). However, when the fusing pass sees an
// already-flattened lowering with the reshape between conv2d and the
// activation, it must still match. This file exercises both shapes.

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_in_4d = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <1x1>, memref<16x32x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_w = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 256 + d1 + d2, d3), <1x1>, memref<262144x1xf32, #system_memory>>
#ttnn_layout_out_4d = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 512 + d1 * 512 + d2, d3), <1x1>, memref<16x32x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_out_3d = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 1024 + d1, d2), <1x1>, memref<32x16x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_perm = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 16384 + d1 * 32 + d2, d3), <1x1>, memref<512x32x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_perm2 = #ttnn.ttnn_layout<(d0, d1, d2, d3) -> (d0 * 524288 + d1 * 512 + d2, d3), <1x1>, memref<16384x1x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>

module {
  ttcore.device_module {
    builtin.module {
      ttcore.device @default_device = <workerGrid = #ttcore.grid<8x8, virt_to_physical_map = (d0, d1) -> (0, d0, d1), physical_to_virt_map = (d0, d1, d2) -> (d1, d2)>, dramGrid = #ttcore.grid<1x12>, l1Map = (d0, d1, d2)[s0] -> (0, d0, d1, d2 + s0), dramMap = (d0, d1, d2)[s0, s1, s2, s3, s4, s5, s6] -> (0, 0, 0, 0), meshShape = , chipIds = [0]>

      // Case 1: conv1d lowering with ONE reshape (4D->3D) between conv2d and ReLU.
      // Expected: fusion fires; conv2d picks up `activation = <op_type = relu>`
      // and the standalone ttnn.relu disappears. The reshape is preserved.
      // CHECK-LABEL: func.func @conv1d_relu_via_reshape
      func.func @conv1d_relu_via_reshape(%arg0: tensor<1x1x512x256xf32, #ttnn_layout_in_4d>, %arg1: tensor<1024x256x1x1xf32, #ttnn_layout_w>) -> tensor<1x1024x512xf32, #ttnn_layout_out_3d> {
        // CHECK: %[[CONV:.*]] = "ttnn.conv2d"
        // CHECK-SAME: activation = <op_type = relu>
        // CHECK-NOT: ttnn.relu
        // CHECK: "ttnn.reshape"(%[[CONV]])
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.conv2d"(%arg0, %arg1, %0) <{batch_size = 1 : i32, dilation = array<i32: 1, 1>, dtype = #ttcore.supportedDataTypes<f32>, groups = 1 : i32, in_channels = 256 : i32, input_height = 512 : i32, input_width = 1 : i32, kernel_size = array<i32: 1, 1>, out_channels = 1024 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x1x512x256xf32, #ttnn_layout_in_4d>, tensor<1024x256x1x1xf32, #ttnn_layout_w>, !ttnn.device) -> tensor<1x1x512x1024xf32, #ttnn_layout_out_4d>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1024 : i32, 512 : i32]}> : (tensor<1x1x512x1024xf32, #ttnn_layout_out_4d>) -> tensor<1x1024x512xf32, #ttnn_layout_out_3d>
        %3 = "ttnn.relu"(%2) : (tensor<1x1024x512xf32, #ttnn_layout_out_3d>) -> tensor<1x1024x512xf32, #ttnn_layout_out_3d>
        return %3 : tensor<1x1024x512xf32, #ttnn_layout_out_3d>
      }

      // Case 2: same shape with GELU (Whisper conv1d->gelu->conv1d->gelu pattern).
      // Pilot 3.1 registered TTNNConv2dWithActivation<GeluOp>; this confirms the
      // registration flows through the reshape-peeling matcher.
      // CHECK-LABEL: func.func @conv1d_gelu_via_reshape
      func.func @conv1d_gelu_via_reshape(%arg0: tensor<1x1x512x256xf32, #ttnn_layout_in_4d>, %arg1: tensor<1024x256x1x1xf32, #ttnn_layout_w>) -> tensor<1x1024x512xf32, #ttnn_layout_out_3d> {
        // CHECK: %[[CONV:.*]] = "ttnn.conv2d"
        // CHECK-SAME: activation = <op_type = gelu>
        // CHECK-NOT: ttnn.gelu
        // CHECK: "ttnn.reshape"(%[[CONV]])
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.conv2d"(%arg0, %arg1, %0) <{batch_size = 1 : i32, dilation = array<i32: 1, 1>, dtype = #ttcore.supportedDataTypes<f32>, groups = 1 : i32, in_channels = 256 : i32, input_height = 512 : i32, input_width = 1 : i32, kernel_size = array<i32: 1, 1>, out_channels = 1024 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x1x512x256xf32, #ttnn_layout_in_4d>, tensor<1024x256x1x1xf32, #ttnn_layout_w>, !ttnn.device) -> tensor<1x1x512x1024xf32, #ttnn_layout_out_4d>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 1024 : i32, 512 : i32]}> : (tensor<1x1x512x1024xf32, #ttnn_layout_out_4d>) -> tensor<1x1024x512xf32, #ttnn_layout_out_3d>
        %3 = "ttnn.gelu"(%2) : (tensor<1x1024x512xf32, #ttnn_layout_out_3d>) -> tensor<1x1024x512xf32, #ttnn_layout_out_3d>
        return %3 : tensor<1x1024x512xf32, #ttnn_layout_out_3d>
      }

      // Case 3: control — direct conv2d -> relu (no reshape) still fuses.
      // CHECK-LABEL: func.func @conv2d_relu_direct
      func.func @conv2d_relu_direct(%arg0: tensor<1x1x512x256xf32, #ttnn_layout_in_4d>, %arg1: tensor<1024x256x1x1xf32, #ttnn_layout_w>) -> tensor<1x1x512x1024xf32, #ttnn_layout_out_4d> {
        // CHECK: "ttnn.conv2d"
        // CHECK-SAME: activation = <op_type = relu>
        // CHECK-NOT: ttnn.relu
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.conv2d"(%arg0, %arg1, %0) <{batch_size = 1 : i32, dilation = array<i32: 1, 1>, dtype = #ttcore.supportedDataTypes<f32>, groups = 1 : i32, in_channels = 256 : i32, input_height = 512 : i32, input_width = 1 : i32, kernel_size = array<i32: 1, 1>, out_channels = 1024 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x1x512x256xf32, #ttnn_layout_in_4d>, tensor<1024x256x1x1xf32, #ttnn_layout_w>, !ttnn.device) -> tensor<1x1x512x1024xf32, #ttnn_layout_out_4d>
        %2 = "ttnn.relu"(%1) : (tensor<1x1x512x1024xf32, #ttnn_layout_out_4d>) -> tensor<1x1x512x1024xf32, #ttnn_layout_out_4d>
        return %2 : tensor<1x1x512x1024xf32, #ttnn_layout_out_4d>
      }

      // Case 4: negative — the matcher does NOT peel through a reshape -> permute -> reshape
      // chain. This shape can appear if --ttir-flatten-sliding-window leaves both the
      // pre-conv and post-conv permute/reshape sandwich on the conv2d boundary AND the
      // activation lives below the rank-recovery sequence. Documented as a known gap;
      // the activation should remain untouched.
      // CHECK-LABEL: func.func @conv1d_relu_chain_no_fuse
      func.func @conv1d_relu_chain_no_fuse(%arg0: tensor<1x1x512x256xf32, #ttnn_layout_in_4d>, %arg1: tensor<1024x256x1x1xf32, #ttnn_layout_w>) -> tensor<1x1024x512xf32, #ttnn_layout_out_3d> {
        // CHECK: %[[CONV:.*]] = "ttnn.conv2d"
        // CHECK-NOT: activation = <op_type
        // CHECK: "ttnn.reshape"(%[[CONV]])
        // CHECK: "ttnn.permute"
        // CHECK: "ttnn.reshape"
        // CHECK: "ttnn.relu"
        %0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
        %1 = "ttnn.conv2d"(%arg0, %arg1, %0) <{batch_size = 1 : i32, dilation = array<i32: 1, 1>, dtype = #ttcore.supportedDataTypes<f32>, groups = 1 : i32, in_channels = 256 : i32, input_height = 512 : i32, input_width = 1 : i32, kernel_size = array<i32: 1, 1>, out_channels = 1024 : i32, padding = array<i32: 0, 0, 0, 0>, stride = array<i32: 1, 1>}> : (tensor<1x1x512x256xf32, #ttnn_layout_in_4d>, tensor<1024x256x1x1xf32, #ttnn_layout_w>, !ttnn.device) -> tensor<1x1x512x1024xf32, #ttnn_layout_out_4d>
        %2 = "ttnn.reshape"(%1) <{shape = [1 : i32, 512 : i32, 1 : i32, 1024 : i32]}> : (tensor<1x1x512x1024xf32, #ttnn_layout_out_4d>) -> tensor<1x512x1x1024xf32, #ttnn_layout_perm>
        %3 = "ttnn.permute"(%2) <{permutation = array<i64: 0, 3, 1, 2>}> : (tensor<1x512x1x1024xf32, #ttnn_layout_perm>) -> tensor<1x1024x512x1xf32, #ttnn_layout_perm2>
        %4 = "ttnn.reshape"(%3) <{shape = [1 : i32, 1024 : i32, 512 : i32]}> : (tensor<1x1024x512x1xf32, #ttnn_layout_perm2>) -> tensor<1x1024x512xf32, #ttnn_layout_out_3d>
        %5 = "ttnn.relu"(%4) : (tensor<1x1024x512xf32, #ttnn_layout_out_3d>) -> tensor<1x1024x512xf32, #ttnn_layout_out_3d>
        return %5 : tensor<1x1024x512xf32, #ttnn_layout_out_3d>
      }
    }
  }
}
