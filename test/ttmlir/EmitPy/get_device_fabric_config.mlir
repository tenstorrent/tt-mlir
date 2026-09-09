// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Verifies that the EmitPy lowering of `ttnn.get_device` translates the correct
// `fabric_config=...` kwarg on the generated `utils.DeviceGetter.get_device`
// call.

// 1. Both axes are ring -> FABRIC_1D_RING.
// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="mesh-shape=2,4 mesh-topology=ring,ring" %s \
// RUN:   | ttmlir-translate --mlir-to-python \
// RUN:   | FileCheck %s --check-prefix=RING-RING

// 2. Only one axis is a ring -> still FABRIC_1D_RING (FABRIC_1D would not be
//    able to route the ring axis).
// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="mesh-shape=2,4 mesh-topology=ring,linear" %s \
// RUN:   | ttmlir-translate --mlir-to-python \
// RUN:   | FileCheck %s --check-prefix=RING-LINEAR

// 3. No ring axes on a multi-device mesh -> FABRIC_1D.
// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="mesh-shape=2,4 mesh-topology=linear,linear" %s \
// RUN:   | ttmlir-translate --mlir-to-python \
// RUN:   | FileCheck %s --check-prefix=LINEAR-LINEAR

// 4. Unit mesh (prod(mesh_shape) == 1) -> DISABLED, regardless of topology.
// RUN: ttmlir-opt --ttir-to-emitpy-pipeline="mesh-shape=1,1 mesh-topology=linear,linear" %s \
// RUN:   | ttmlir-translate --mlir-to-python \
// RUN:   | FileCheck %s --check-prefix=UNIT-MESH

func.func @add(%arg0: tensor<32x32xf32>, %arg1: tensor<32x32xf32>) -> tensor<32x32xf32> {
  %0 = "ttir.add"(%arg0, %arg1) : (tensor<32x32xf32>, tensor<32x32xf32>) -> tensor<32x32xf32>
  return %0 : tensor<32x32xf32>
}

// RING-RING:     utils.DeviceGetter.get_device({{.*}}fabric_config=ttnn.FabricConfig.FABRIC_1D_RING)

// RING-LINEAR:   utils.DeviceGetter.get_device({{.*}}fabric_config=ttnn.FabricConfig.FABRIC_1D_RING)

// LINEAR-LINEAR: utils.DeviceGetter.get_device({{.*}}fabric_config=ttnn.FabricConfig.FABRIC_1D)

// UNIT-MESH:     utils.DeviceGetter.get_device({{.*}}fabric_config=ttnn.FabricConfig.DISABLED)
