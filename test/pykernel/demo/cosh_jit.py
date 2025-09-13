# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pykernel
import ttnn


def _cosh(a):
    e_pos_x = exp(a)
    e_neg_x = exp(-a)
    nr_term = e_pos_x + e_neg_x
    return nr_term * 0.5


def main():
    input_tensor = ttnn.Tensor(
        [-1, 0, 1, 2],
        shape=[1, 1, 1, 4],
        data_type=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
    )

    cosh_ttnn = pykernel.jit(backend="ttnn", debug=True, dump_flatbuffer=True)(_cosh)
    cosh_metal = pykernel.jit(backend="metal", debug=True, dump_flatbuffer=True)(_cosh)

    cosh_ttnn(input_tensor)
    cosh_metal(input_tensor)


if __name__ == "__main__":
    main()
