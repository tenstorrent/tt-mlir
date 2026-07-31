// SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
func.func @gather_rows(%w: tensor<256x128xf32>, %ix: tensor<256x128xi32>) -> tensor<256x128xf32> {
  %f = "ttir.typecast"(%ix) : (tensor<256x128xi32>) -> tensor<256x128xf32>
  %z = "ttir.multiply"(%f, %f) : (tensor<256x128xf32>, tensor<256x128xf32>) -> tensor<256x128xf32>
  %o = "ttir.abs"(%w) : (tensor<256x128xf32>) -> tensor<256x128xf32>
  return %o : tensor<256x128xf32>
}
