// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt --ttnn-prepare-d2m-subgraphs-for-trace -o %t %s
// RUN: FileCheck %s --input-file=%t

// Unit tests for TTNNPrepareD2MSubgraphsForTrace. The pass runs on
// forward-device functions and, ahead of trace hoisting, (1) merges all
// ttnn.get_device ops into a single one at the top of the block and (2) moves
// all ttnn.empty scratch buffers into the prelude right after that device. The
// ttnn.add ops below stand in for the surrounding hoistable ops / D2M generics;
// the pass only relocates the get_device and empty ops.

#l = #ttnn.ttnn_layout<(d0, d1, d2) -> (d0 * 32 + d1, d2), <1x1>, memref<1x1x!ttcore.tile<32x32, f32>, #ttnn.buffer_type<dram>>, <interleaved>>

module {
  // Two subgraphs, each with its own get_device + empty interleaved in the
  // middle. Expect: one get_device at the top, both (distinct) empties right
  // after it, the duplicate get_device gone, and the original op order of the
  // rest preserved.
  // CHECK-LABEL: func.func @interleaved_subgraphs
  // CHECK-NEXT:    %[[DEV:.+]] = "ttnn.get_device"
  // CHECK-NEXT:    %[[E0:.+]] = "ttnn.empty"(%[[DEV]])
  // CHECK-NEXT:    %[[E1:.+]] = "ttnn.empty"(%[[DEV]])
  // CHECK-NEXT:    %[[A0:.+]] = "ttnn.add"(%arg0, %arg1)
  // CHECK-NEXT:    %[[A1:.+]] = "ttnn.add"(%[[A0]], %[[E0]])
  // CHECK-NEXT:    %[[A2:.+]] = "ttnn.add"(%[[A1]], %[[E1]])
  // CHECK-NEXT:    return %[[A2]]
  func.func @interleaved_subgraphs(%arg0: tensor<1x17x32xf32, #l>, %arg1: tensor<1x17x32xf32, #l>) -> tensor<1x17x32xf32, #l> attributes {tt.function_type = "forward_device"} {
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<1x17x32xf32, #l>, tensor<1x17x32xf32, #l>) -> tensor<1x17x32xf32, #l>
    %d0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %e0 = "ttnn.empty"(%d0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<1x17x32>}> : (!ttnn.device) -> tensor<1x17x32xf32, #l>
    %1 = "ttnn.add"(%0, %e0) : (tensor<1x17x32xf32, #l>, tensor<1x17x32xf32, #l>) -> tensor<1x17x32xf32, #l>
    %d1 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %e1 = "ttnn.empty"(%d1) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<1x17x32>}> : (!ttnn.device) -> tensor<1x17x32xf32, #l>
    %2 = "ttnn.add"(%1, %e1) : (tensor<1x17x32xf32, #l>, tensor<1x17x32xf32, #l>) -> tensor<1x17x32xf32, #l>
    return %2 : tensor<1x17x32xf32, #l>
  }

  // Single subgraph with the device mid-block: nothing to dedup, but the
  // device and its empty are still moved up into the prelude.
  // CHECK-LABEL: func.func @single_subgraph
  // CHECK-NEXT:    %[[DEV:.+]] = "ttnn.get_device"
  // CHECK-NEXT:    %[[E:.+]] = "ttnn.empty"(%[[DEV]])
  // CHECK-NEXT:    %[[A0:.+]] = "ttnn.add"(%arg0, %arg0)
  // CHECK-NEXT:    %[[A1:.+]] = "ttnn.add"(%[[A0]], %[[E]])
  // CHECK-NEXT:    return %[[A1]]
  func.func @single_subgraph(%arg0: tensor<1x17x32xf32, #l>) -> tensor<1x17x32xf32, #l> attributes {tt.function_type = "forward_device"} {
    %0 = "ttnn.add"(%arg0, %arg0) : (tensor<1x17x32xf32, #l>, tensor<1x17x32xf32, #l>) -> tensor<1x17x32xf32, #l>
    %d0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %e0 = "ttnn.empty"(%d0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<1x17x32>}> : (!ttnn.device) -> tensor<1x17x32xf32, #l>
    %1 = "ttnn.add"(%0, %e0) : (tensor<1x17x32xf32, #l>, tensor<1x17x32xf32, #l>) -> tensor<1x17x32xf32, #l>
    return %1 : tensor<1x17x32xf32, #l>
  }

  // No get_device: the pass should early-return and leave the function
  // untouched (first op stays an add, no device injected).
  // CHECK-LABEL: func.func @no_device
  // CHECK-NEXT:    %[[A0:.+]] = "ttnn.add"(%arg0, %arg1)
  // CHECK-NEXT:    %[[A1:.+]] = "ttnn.add"(%[[A0]], %arg0)
  // CHECK-NEXT:    return %[[A1]]
  func.func @no_device(%arg0: tensor<1x17x32xf32, #l>, %arg1: tensor<1x17x32xf32, #l>) -> tensor<1x17x32xf32, #l> attributes {tt.function_type = "forward_device"} {
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<1x17x32xf32, #l>, tensor<1x17x32xf32, #l>) -> tensor<1x17x32xf32, #l>
    %1 = "ttnn.add"(%0, %arg0) : (tensor<1x17x32xf32, #l>, tensor<1x17x32xf32, #l>) -> tensor<1x17x32xf32, #l>
    return %1 : tensor<1x17x32xf32, #l>
  }

  // Not a forward-device function: the pass must skip it, so the interleaved
  // get_device/empty stay exactly where they were (device still after the add).
  // CHECK-LABEL: func.func @not_forward_device
  // CHECK-NEXT:    %[[A0:.+]] = "ttnn.add"(%arg0, %arg1)
  // CHECK-NEXT:    "ttnn.get_device"
  // CHECK-NEXT:    "ttnn.empty"
  // CHECK-NEXT:    "ttnn.add"
  // CHECK-NEXT:    return
  func.func @not_forward_device(%arg0: tensor<1x17x32xf32, #l>, %arg1: tensor<1x17x32xf32, #l>) -> tensor<1x17x32xf32, #l> {
    %0 = "ttnn.add"(%arg0, %arg1) : (tensor<1x17x32xf32, #l>, tensor<1x17x32xf32, #l>) -> tensor<1x17x32xf32, #l>
    %d0 = "ttnn.get_device"() <{mesh_offset = #ttnn<mesh_offset 0x0>, mesh_shape = #ttnn<mesh_shape 1x1>}> : () -> !ttnn.device
    %e0 = "ttnn.empty"(%d0) <{layout = #ttnn.layout<tile>, shape = #ttnn.shape<1x17x32>}> : (!ttnn.device) -> tensor<1x17x32xf32, #l>
    %1 = "ttnn.add"(%0, %e0) : (tensor<1x17x32xf32, #l>, tensor<1x17x32xf32, #l>) -> tensor<1x17x32xf32, #l>
    return %1 : tensor<1x17x32xf32, #l>
  }
}
