# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import ttnn_jit
import ttnn
import torch


def _cosh(a):
    e_pos_x = exp(a)
    e_neg_x = exp(-a)
    nr_term = e_pos_x + e_neg_x
    return nr_term * 0.5


def main():
    device = ttnn.open_device(device_id=0)
    input_tensor = ttnn.Tensor(
        [-1, 0, 1, 2],
        shape=[1, 1, 1, 4],
        data_type=ttnn.float32,
        layout=ttnn.TILE_LAYOUT,
        device=device,
    )

    # cosh_metal = ttnn_jit.jit(backend="metal", debug=True, dump_flatbuffer=True)(_cosh)
    # cosh_metal(input_tensor)

    cosh_ttnn = ttnn_jit.jit(backend="ttnn", debug=True, dump_flatbuffer=True)(_cosh)
    output_tensor = cosh_ttnn(input_tensor)
    golden = torch.cosh(input_tensor.cpu().to_torch())

    print("input_tensor: ", input_tensor)
    print("output_tensor: ", output_tensor)
    print("golden: ", golden)

    all_close = torch.allclose(output_tensor.cpu().to_torch(), golden, atol=1e-2)
    print("all_close: ", all_close)

    ttnn.close_device(device)


if __name__ == "__main__":
    main()
