// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// RUN: ttmlir-opt %s --canonicalize | FileCheck %s

module attributes {ttcore.meshes = #ttcore.meshes<[<"mesh" = 1x2>]>} {
  func.func @preserve_mesh_annotation(%arg0: tensor<32x32xbf16>) -> tensor<32x32xbf16, #ttcore.tensor_mesh<"mesh">> {
    %0 = d2m.mesh_shard %arg0 {shard_dims = array<i64: -1, -1>, shard_direction = #ttcore.shard_direction<full_to_shard>, shard_shape = array<i64: 1, 1>, shard_type = #ttcore.shard_type<devices>} : tensor<32x32xbf16> -> tensor<32x32xbf16, #ttcore.tensor_mesh<"mesh">>
    return %0 : tensor<32x32xbf16, #ttcore.tensor_mesh<"mesh">>
  }
}

// CHECK-LABEL: func.func @preserve_mesh_annotation
// CHECK: d2m.mesh_shard
