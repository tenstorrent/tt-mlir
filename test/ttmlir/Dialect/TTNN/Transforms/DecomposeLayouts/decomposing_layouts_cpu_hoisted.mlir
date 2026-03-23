// SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttcore-register-device="system-desc-path=%system_desc_path%" --ttnn-decompose-layouts -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that CPU-hoisted function outputs are tilized/typecast on host before
// transfer to device, and CPU-hoisted function inputs are untilized/typecast on
// host after transfer from device.

#dram = #ttnn.buffer_type<dram>
#system_memory = #ttnn.buffer_type<system_memory>
#ttnn_layout_host_rm = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xf32, #system_memory>>
#ttnn_layout_device_rm_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<64x128xbf16, #dram>, <interleaved>>
#ttnn_layout_device_tile = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, f32>, #dram>, <interleaved>>
#ttnn_layout_device_tile_bf16 = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<2x4x!ttcore.tile<32x32, bf16>, #dram>, <interleaved>>

module attributes {} {
  func.func private @cpu_hoisted_rm() -> tensor<64x128xf32, #ttnn_layout_host_rm>
  func.func private @cpu_hoisted_input_rm(%arg0: tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_host_rm>

  // ToLayout(tile on host) -> ToDevice
  func.func @cpu_hoisted_layout_no_typecast_rm_to_tile() -> tensor<64x128xf32, #ttnn_layout_device_tile> {
    // CHECK-LABEL: func.func @cpu_hoisted_layout_no_typecast_rm_to_tile
    // CHECK: %[[GET_DEVICE:.*]] = "ttnn.get_device"()
    // CHECK: %[[CALL:.*]] = call @cpu_hoisted_rm()
    // CHECK-NEXT: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%[[CALL]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-NEXT: %[[TO_DEVICE:.*]] = "ttnn.to_device"(%[[TO_LAYOUT]], %[[GET_DEVICE]])
    // CHECK-NOT: "ttnn.to_layout"
    // CHECK: return %[[TO_DEVICE]]
    %0 = func.call @cpu_hoisted_rm() {ttir.cpu_hoist_call = unit} : () -> tensor<64x128xf32, #ttnn_layout_host_rm>
    %1 = "ttnn.to_layout"(%0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_device_tile>
    return %1 : tensor<64x128xf32, #ttnn_layout_device_tile>
  }

  // TypecastOp(host) -> ToDevice
  func.func @cpu_hoisted_no_layout_typecast_rm_f32_to_rm_bf16() -> tensor<64x128xbf16, #ttnn_layout_device_rm_bf16> {
    // CHECK-LABEL: func.func @cpu_hoisted_no_layout_typecast_rm_f32_to_rm_bf16
    // CHECK: %[[GET_DEVICE:.*]] = "ttnn.get_device"()
    // CHECK: %[[CALL:.*]] = call @cpu_hoisted_rm()
    // CHECK-NEXT: %[[TYPECAST:.*]] = "ttnn.typecast"(%[[CALL]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-NEXT: %[[TO_DEVICE:.*]] = "ttnn.to_device"(%[[TYPECAST]], %[[GET_DEVICE]])
    // CHECK-NOT: "ttnn.typecast"
    // CHECK: return %[[TO_DEVICE]]
    %0 = func.call @cpu_hoisted_rm() {ttir.cpu_hoist_call = unit} : () -> tensor<64x128xf32, #ttnn_layout_host_rm>
    %1 = "ttnn.to_layout"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xbf16, #ttnn_layout_device_rm_bf16>
    return %1 : tensor<64x128xbf16, #ttnn_layout_device_rm_bf16>
  }

  // TypecastOp(host) -> ToLayout(tile on host) -> ToDevice
  func.func @cpu_hoisted_layout_typecast_rm_f32_to_tile_bf16() -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16> {
    // CHECK-LABEL: func.func @cpu_hoisted_layout_typecast_rm_f32_to_tile_bf16
    // CHECK: %[[GET_DEVICE:.*]] = "ttnn.get_device"()
    // CHECK: %[[CALL:.*]] = call @cpu_hoisted_rm()
    // CHECK-NEXT: %[[TYPECAST:.*]] = "ttnn.typecast"(%[[CALL]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<bf16>
    // CHECK-NEXT: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%[[TYPECAST]])
    // CHECK-SAME: layout = #ttnn.layout<tile>
    // CHECK-NEXT: %[[TO_DEVICE:.*]] = "ttnn.to_device"(%[[TO_LAYOUT]], %[[GET_DEVICE]])
    // CHECK: return %[[TO_DEVICE]]
    %0 = func.call @cpu_hoisted_rm() {ttir.cpu_hoist_call = unit} : () -> tensor<64x128xf32, #ttnn_layout_host_rm>
    %1 = "ttnn.to_layout"(%0) <{dtype = #ttcore.supportedDataTypes<bf16>, layout = #ttnn.layout<tile>, memory_config = #ttnn.memory_config<#dram, <interleaved>>}> : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
    return %1 : tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>
  }

  // FromDevice -> ToLayout(rm on host)
  func.func @cpu_hoisted_input_layout_no_typecast_tile_to_rm(%arg0: tensor<64x128xf32, #ttnn_layout_device_tile>) -> tensor<64x128xf32, #ttnn_layout_host_rm> {
    // CHECK-LABEL: func.func @cpu_hoisted_input_layout_no_typecast_tile_to_rm
    // CHECK: %[[FROM_DEVICE:.*]] = "ttnn.from_device"(%arg0)
    // CHECK-NEXT: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%[[FROM_DEVICE]])
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-NEXT: %[[CALL:.*]] = call @cpu_hoisted_input_rm(%[[TO_LAYOUT]])
    // CHECK-NOT: "ttnn.to_layout"
    // CHECK: return %[[CALL]]
    %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<64x128xf32, #ttnn_layout_device_tile>) -> tensor<64x128xf32, #ttnn_layout_host_rm>
    %1 = func.call @cpu_hoisted_input_rm(%0) {ttir.cpu_hoist_call = unit} : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_host_rm>
    return %1 : tensor<64x128xf32, #ttnn_layout_host_rm>
  }

  // FromDevice -> TypecastOp(host) -> ToLayout(rm on host)
  func.func @cpu_hoisted_input_layout_typecast_tile_bf16_to_rm_f32(%arg0: tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>) -> tensor<64x128xf32, #ttnn_layout_host_rm> {
    // CHECK-LABEL: func.func @cpu_hoisted_input_layout_typecast_tile_bf16_to_rm_f32
    // CHECK: %[[FROM_DEVICE:.*]] = "ttnn.from_device"(%arg0)
    // CHECK-NEXT: %[[TYPECAST:.*]] = "ttnn.typecast"(%[[FROM_DEVICE]])
    // CHECK-SAME: dtype = #ttcore.supportedDataTypes<f32>
    // CHECK-NEXT: %[[TO_LAYOUT:.*]] = "ttnn.to_layout"(%[[TYPECAST]])
    // CHECK-SAME: layout = #ttnn.layout<row_major>
    // CHECK-NEXT: %[[CALL:.*]] = call @cpu_hoisted_input_rm(%[[TO_LAYOUT]])
    // CHECK: return %[[CALL]]
    %0 = "ttnn.to_layout"(%arg0) <{dtype = #ttcore.supportedDataTypes<f32>, layout = #ttnn.layout<row_major>, memory_config = #ttnn.memory_config<#system_memory>}> : (tensor<64x128xbf16, #ttnn_layout_device_tile_bf16>) -> tensor<64x128xf32, #ttnn_layout_host_rm>
    %1 = func.call @cpu_hoisted_input_rm(%0) {ttir.cpu_hoist_call = unit} : (tensor<64x128xf32, #ttnn_layout_host_rm>) -> tensor<64x128xf32, #ttnn_layout_host_rm>
    return %1 : tensor<64x128xf32, #ttnn_layout_host_rm>
  }
}
