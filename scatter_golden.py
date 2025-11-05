# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import numpy as np

# This script is used to help write the lowering pass from ttir to d2m
# Test cases are taken from https://openxla.org/stablehlo/spec#scatter


def is_index_in_bounds(tensor, index):
    shape = tensor.shape
    return all(0 <= idx < dim for idx, dim in zip(index, shape))


def scatter(
    inputs: np.array,
    scatter_indices: np.array,
    updates: np.array,
    update_window_dims: list,
    inserted_window_dims: list,
    input_batching_dims: list,
    scatter_indices_batching_dims: list,
    scatter_dims_to_operand_dims: list,
    index_vector_dim: int,
    indices_are_sorted: bool,
    unique_indices: bool,
    update_computation=None,
):
    output = np.copy(inputs)

    for i in range(updates.shape[0]):
        for j in range(updates.shape[1]):
            for k in range(updates.shape[2]):
                for a in range(updates.shape[3]):
                    for b in range(updates.shape[4]):
                        axes = [0, 1, 2, 3, 4]
                        update_index = [i, j, k, a, b]

                        """
                        update_scatter_dims = [d for d in axes(updates[0]) and d not in update_window_dims]
                        """
                        update_scatter_dims = [
                            d for d in axes if d not in update_window_dims
                        ]

                        """
                        update_scatter_index = update_index[update_scatter_dims...]
                        """
                        update_scatter_index = [
                            update_index[d] for d in update_scatter_dims
                        ]

                        """
                        start_index is defined as:
                        scatter_indices[si0, ..., :, ..., siN] where si are individual elements in update_scatter_index and : is inserted at the index_vector_dim index, if index_vector_dim < rank(scatter_indices).
                        [scatter_indices[update_scatter_index]] otherwise.
                        """
                        start_index = None
                        update_index_copy = update_scatter_index.copy()

                        if index_vector_dim < scatter_indices.ndim:
                            index_list = []
                            scatter_len = len(update_index_copy) + 1

                            for dim in range(scatter_len):
                                if dim == index_vector_dim:
                                    index_list.append(slice(None))
                                else:
                                    index_list.append(update_index_copy.pop(0))

                            start_index = scatter_indices[tuple(index_list)]
                        else:
                            start_index = [scatter_indices[update_index_copy]]

                        """
                        For d_input in axes(inputs[0]),
                        full_start_index[d_input] = start_index[d_start] if d_input = scatter_dims_to_operand_dims[d_start].
                        full_start_index[d_input] = 0 otherwise.
                        """
                        full_start_index = [0] * inputs.ndim
                        for scatter_dims in range(len(scatter_dims_to_operand_dims)):
                            full_start_index[
                                scatter_dims_to_operand_dims[scatter_dims]
                            ] = start_index[scatter_dims]

                        """
                        For d_input in axes(inputs[0]),
                        full_batching_index[d_input] = update_scatter_index[d_start - (d_start < index_vector_dim ? 0 : 1)] if d_input = input_batching_dims[i_batching] and d_start = scatter_indices_batching_dims[i_batching].
                        full_batching_index[d_input] = 0 otherwise.
                        """
                        full_batching_index = [0] * inputs.ndim

                        for dim in range(inputs.shape[0]):
                            if dim in input_batching_dims:
                                batching_idx = input_batching_dims.index(dim)
                                start_dim = scatter_indices_batching_dims[batching_idx]

                                if start_dim < index_vector_dim:
                                    index_value = update_scatter_index[start_dim]
                                else:
                                    index_value = update_scatter_index[start_dim - 1]

                                full_batching_index[dim] = index_value
                            else:
                                full_batching_index[dim] = 0

                        """
                        update_window_index = update_index[update_window_dims...].
                        """
                        update_window_index = [
                            update_index[d] for d in update_window_dims
                        ]

                        """
                        full_window_index = [wi0, ..., 0, ..., wiN] where wi are individual elements in update_window_index,
                        and 0 is inserted at indices from inserted_window_dims and input_batching_dims
                        """
                        zero_indices = set(inserted_window_dims) | set(
                            input_batching_dims
                        )
                        total_len = len(update_window_index) + len(zero_indices)

                        full_window_index = [
                            0 if i in zero_indices else update_window_index.pop(0)
                            for i in range(total_len)
                        ]

                        """
                        result_index = full_start_index + full_batching_index + full_window_index
                        """
                        result_index = [
                            int(a) + int(b) + int(c)
                            for a, b, c in zip(
                                full_start_index, full_batching_index, full_window_index
                            )
                        ]

                        if is_index_in_bounds(output, result_index):
                            output[tuple(result_index)] = (
                                updates[i, j, k, a, b] + output[tuple(result_index)]
                            )

    return output


if __name__ == "__main__":
    inputs = [
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

    scatter_indices = [
        [[[0, 0], [1, 0], [2, 1]], [[0, 1], [1, 1], [0, 9]]],
        [[[0, 0], [2, 1], [2, 2]], [[1, 2], [0, 1], [1, 0]]],
    ]

    updates = [
        [
            [[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]],
            [[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]],
        ],
        [
            [[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]],
            [[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]],
        ],
    ]

    golden = [
        [
            [[3, 4], [6, 7], [6, 7], [7, 8]],
            [[9, 10], [11, 12], [15, 16], [17, 18]],
            [[17, 18], [19, 20], [22, 23], [24, 25]],
        ],
        [
            [[25, 26], [28, 29], [30, 31], [31, 32]],
            [[35, 36], [38, 39], [38, 39], [39, 40]],
            [[41, 42], [44, 45], [46, 47], [47, 48]],
        ],
    ]

    inputs_np = np.array(inputs)
    scatter_indices_np = np.array(scatter_indices)
    updates_np = np.array(updates)
    golden_np = np.array(golden)

    update_window_dims = [3, 4]
    inserted_window_dims = [1]
    input_batching_dims = [0]
    scatter_indices_batching_dims = [1]
    scatter_dims_to_operand_dims = [2, 1]
    index_vector_dim = 3
    indices_are_sorted = False
    unique_indices = False

    output = scatter(
        inputs_np,
        scatter_indices_np,
        updates_np,
        update_window_dims,
        inserted_window_dims,
        input_batching_dims,
        scatter_indices_batching_dims,
        scatter_dims_to_operand_dims,
        index_vector_dim,
        indices_are_sorted,
        unique_indices,
    )

    # Verify correctness
    assert np.array_equal(output, golden_np), "Output does not match the golden result!"
