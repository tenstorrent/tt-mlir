// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt \
// RUN:   --ttir-to-ttnn-backend-pipeline="system-desc-path=%system_desc_path%" \
// RUN:   --ttcore-unwrap-device-module \
// RUN:   --ttnn-tuplify-tensors="tuplify-input-if-empty=true" \
// RUN:   --ttnn-prepare-module-for-export \
// RUN:   -o %t %s
// RUN: FileCheck %s --input-file=%t

// Verify that TTNNPrepareModuleForExport reshapes the forward function into
// the canonical entry-point expected on the tt-xla / codegen boundary:
// - the forward function is renamed to `forward`;
// - a trailing `!ttnn.device` argument is injected;
// - each device-bound input tuple element is retyped to host row-major and
//   staged onto the device via `ttnn.to_layout` followed by `ttnn.to_device`;
// - the first and second arguments carry `emitpy.name = "input"` and
//   `"device"` respectively.

// CHECK-DAG: #[[HOST:[a-zA-Z0-9_]+]] = #ttnn.ttnn_layout<(d0, d1) -> (d0, d1), <1x1>, memref<32x32xbf16, #system_memory>>

// CHECK: func.func @forward(
// CHECK-SAME: tuple<tensor<32x32xbf16, #[[HOST]]>, tensor<32x32xbf16, #[[HOST]]>> {emitpy.name = "input"}
// CHECK-SAME: !ttnn.device {emitpy.name = "device"}

// CHECK: %[[E0:.+]] = ttcore.get_tuple_element {{.*}}[0]
// CHECK-SAME: -> tensor<32x32xbf16, #[[HOST]]>
// CHECK-NEXT: %[[L0:.+]] = "ttnn.to_layout"(%[[E0]])
// CHECK-NEXT: %{{.+}} = "ttnn.to_device"(%[[L0]]
//
// CHECK: %[[E1:.+]] = ttcore.get_tuple_element {{.*}}[1]
// CHECK-SAME: -> tensor<32x32xbf16, #[[HOST]]>
// CHECK-NEXT: %[[L1:.+]] = "ttnn.to_layout"(%[[E1]])
// CHECK-NEXT: %{{.+}} = "ttnn.to_device"(%[[L1]]

module {
  func.func @add(%arg0: tensor<32x32xbf16>, %arg1: tensor<32x32xbf16>) -> tensor<32x32xbf16> {
    %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xbf16>, tensor<32x32xbf16>) -> tensor<32x32xbf16>
    return %0 : tensor<32x32xbf16>
  }
}
