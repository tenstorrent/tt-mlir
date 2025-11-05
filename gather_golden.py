# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np

# In order to determine output shape, you can use algorithm from: https://github.com/openxla/stablehlo/blob/1ef9e390b5295e676d2b864fe1924bc2f3f4cf0f/stablehlo/dialect/TypeInference.cpp#L2817
# However for our purpose (and conversion in tt-mlir), the output shape is already inferred, thus we pass it as a parameter
# This script is used to help write the lowering pass from ttir to d2m
# Test cases are taken from https://openxla.org/stablehlo/spec#gather


def gather(
    operand,
    start_indices,
    offset_dims,
    collapsed_slice_dims,
    operand_batching_dims,
    start_indices_batching_dims,
    start_index_map,
    index_vector_dim,
    slice_sizes,
    indices_are_sorted,
    output_shape,
):
    output = np.empty(output_shape)

    for i in range(output.shape[0]):
        for j in range(output.shape[1]):
            for k in range(output.shape[2]):
                for a in range(output.shape[3]):
                    for b in range(output.shape[4]):
                        axes = [0, 1, 2, 3, 4]
                        output_index = [i, j, k, a, b]

                        """
                        batch_dims = [d for d in axes(result) and d not in offset_dims]
                        """
                        batch_dims = [d for d in axes if d not in offset_dims]

                        """
                        batch_index = result_index[batch_dims...]
                        """
                        batch_index = [output_index[d] for d in batch_dims]

                        """
                        start_index is defined as:
                        start_indices[bi0, ..., :, ..., biN] where bi are individual elements in batch_index and : is inserted at the index_vector_dim index, if index_vector_dim < rank(start_indices).
                        [start_indices[batch_index]] otherwise.
                        """
                        start_index = None
                        batch_index_copy = batch_index.copy()

                        if index_vector_dim < start_indices.ndim:
                            index_list = []
                            start_len = len(batch_index_copy) + 1

                            for dim in range(start_len):
                                if dim == index_vector_dim:
                                    index_list.append(slice(None))
                                else:
                                    index_list.append(batch_index_copy.pop(0))

                            start_index = start_indices[tuple(index_list)]
                        else:
                            start_index = [start_indices[batch_index_copy]]

                        """
                        For d_operand in axes(operand),
                        full_start_index[d_operand] = clamp(start_index[d_start], 0, dim(operand, d_operand) - slice_sizes[d_operand]) if d_operand = start_index_map[d_start].
                        full_start_index[d_operand] = 0 otherwise.
                        """
                        full_start_index = [0] * operand.ndim
                        for d_start in range(len(start_index_map)):
                            d_operand = start_index_map[d_start]
                            clamped_index = max(
                                0,
                                min(
                                    start_index[d_start],
                                    operand.shape[d_operand] - slice_sizes[d_operand],
                                ),
                            )
                            full_start_index[d_operand] = clamped_index

                        """
                        For d_operand in axes(operand),
                        full_batching_index[d_operand] = batch_index[d_start - (d_start < index_vector_dim ? 0 : 1)] if d_operand = operand_batching_dims[i_batching] and d_start = start_indices_batching_dims[i_batching].
                        full_batching_index[d_operand] = 0 otherwise.
                        """
                        full_batching_index = [0] * operand.ndim
                        for dim in range(operand.shape[0]):
                            if dim in operand_batching_dims:
                                batching_idx = operand_batching_dims.index(dim)
                                start_dim = start_indices_batching_dims[batching_idx]

                                if start_dim < index_vector_dim:
                                    index_value = batch_index[start_dim]
                                else:
                                    index_value = batch_index[start_dim - 1]

                                full_batching_index[dim] = index_value
                            else:
                                full_batching_index[dim] = 0

                        """
                        offset_index = result_index[offset_dims...]
                        """
                        offset_index = [output_index[d] for d in offset_dims]

                        """
                        full_offset_index = [oi0, ..., 0, ..., oiN] where oi are individual elements in offset_index, and 0 is inserted at indices from collapsed_slice_dims and operand_batching_dims.
                        """
                        zero_offset_indices = set(collapsed_slice_dims) | set(
                            operand_batching_dims
                        )
                        total_offset_len = len(offset_index) + len(zero_offset_indices)

                        full_offset_index = [
                            0 if d in zero_offset_indices else offset_index.pop(0)
                            for d in range(total_offset_len)
                        ]

                        """
                        operand_index = full_start_index + full_batching_index + full_offset_index
                        """
                        operand_index = [
                            full_start_index[d]
                            + full_batching_index[d]
                            + full_offset_index[d]
                            for d in range(operand.ndim)
                        ]

                        output[i, j, k, a, b] = operand[tuple(operand_index)]

    return output


if __name__ == "__main__":
    operand = [
        [
            [[1, 2], [3, 4], [5, 6], [7, 8]],
            [[9, 10], [11, 12], [13, 14], [15, 16]],
            [[17, 18], [19, 20], [21, 22], [23, 24]],
        ],
        [
            [[25, 26], [27, 28], [29, 30], [31, 32]],
            [[33, 34], [35, 36], [37, 38], [39, 40]],
            [[41, 42], [43, 44], [45, 46], [47, 48]],
        ],
    ]

    start_indices = [
        [[[0, 0], [1, 0], [2, 1]], [[0, 1], [1, 1], [0, 9]]],
        [[[0, 0], [2, 1], [2, 2]], [[1, 2], [0, 1], [1, 0]]],
    ]

    golden = [
        [
            [[[1, 2], [3, 4]], [[3, 4], [5, 6]], [[13, 14], [15, 16]]],
            [[[33, 34], [35, 36]], [[35, 36], [37, 38]], [[41, 42], [43, 44]]],
        ],
        [
            [[[1, 2], [3, 4]], [[13, 14], [15, 16]], [[21, 22], [23, 24]]],
            [[[43, 44], [45, 46]], [[33, 34], [35, 36]], [[27, 28], [29, 30]]],
        ],
    ]

    operand_np = np.array(operand)
    start_indices_np = np.array(start_indices)
    golden_np = np.array(golden)

    offset_dims = [3, 4]
    collapsed_slice_dims = [1]
    operand_batching_dims = [0]
    start_indices_batching_dims = [1]
    start_index_map = [2, 1]
    index_vector_dim = 3
    slice_sizes = [1, 1, 2, 2]
    indices_are_sorted = False

    output = gather(
        operand_np,
        start_indices_np,
        offset_dims,
        collapsed_slice_dims,
        operand_batching_dims,
        start_indices_batching_dims,
        start_index_map,
        index_vector_dim,
        slice_sizes,
        indices_are_sorted,
        golden_np.shape,
    )

    golden_np = np.array(golden)
    assert np.array_equal(output, golden_np), "Output does not match the golden result!"
