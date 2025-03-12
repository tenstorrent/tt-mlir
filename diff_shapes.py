# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
Hs, Ws = [50, 128, 224, 960], [50, 128, 224, 540]


def permute(cnt, h, w):
    return f"""
func.func @forward_{cnt}(%arg0: tensor<1x1x{h}x{w}xbf16>) -> tensor<1x1x{w}x{h}xbf16> {{
  // CHECK: "ttnn.permute"
  %0 = "ttnn.permute"(%arg0) <{{permutation = array<i64: 0, 1, 3, 2>}}> : (tensor<1x1x{h}x{w}xbf16>) -> tensor<1x1x{w}x{h}xbf16>
  return %0 : tensor<1x1x{w}x{h}xbf16>
}}
"""


i = 0
for h in Hs:
    for w in Ws:
        print(permute(i, h, w))
        i += 1
        print(permute(i, h, w))  # because each permute occurs in the test twice
        i += 1
