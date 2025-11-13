"""
stablehlo_scatter_sim.py

Simulates StableHLO scatter semantics (generic) by iterating over all indices
of `updates` and applying the per-element mapping to the result tensor.

Requires: numpy
"""

import numpy as np
from typing import Sequence, Callable, Tuple

def scatter_sim(
    inputs: np.ndarray,
    scatter_indices: np.ndarray,
    updates: np.ndarray,
    *,
    update_window_dims: Sequence[int],
    inserted_window_dims: Sequence[int],
    input_batching_dims: Sequence[int],
    scatter_indices_batching_dims: Sequence[int],
    scatter_dims_to_operand_dims: Sequence[int],
    index_vector_dim: int,
    indices_are_sorted: bool = False,   # not used in this simulation
    unique_indices: bool = False,       # not used in this simulation
    update_computation: Callable[[np.ndarray, np.ndarray], np.ndarray] = None
) -> np.ndarray:
    """
    Perform a scatter as described by StableHLO semantics by iterating all update indices.
    - inputs: base tensor (will not be mutated; a copy is returned).
    - scatter_indices: tensor describing start indices (contains integer index vectors).
    - updates: update values tensor whose elements will be mapped into result.
    - update_window_dims etc.: attributes from stablehlo.scatter.
    - update_computation: function (destination_value, update_value) -> new_value
                        default is elementwise add for scalars.
    """

    '''
    inputs shape: (2, 3, 4, 2)
    scatter_indices shape: (2, 2, 3, 2)
    updates shape: (2, 2, 3, 2, 2)
    result shape: (2, 3, 4, 2)
    input rank: 4
    scatter_indices rank: 4
    update rank: 5
    result rank: 4

    update_window_dims: [3, 4]
    inserted_window_dims: [1]
    input_batching_dims: [0]
    scatter_indices_batching_dims: [1]
    scatter_dims_to_operand_dims: [2, 1]
    index_vector_dim: 3

    update_scatter_dims: [0, 1, 2]
    '''


    if update_computation is None:
        # default: numeric add (works for elementwise scalars)
        update_computation = lambda a, b: a + b

    # copy inputs to start result
    result = inputs.copy()

    # helper
    def in_bounds(idx: Tuple[int, ...], shape: Sequence[int]) -> bool:
        return all(0 <= idx[d] < shape[d] for d in range(len(shape)))

    # Precompute dims sets & ranks
    input_rank = inputs.ndim
    scatter_idx_rank = scatter_indices.ndim
    update_rank = updates.ndim

    print("input rank:", input_rank) 
    print("scatter_indices rank:", scatter_idx_rank)
    print("update rank:", update_rank)
    print("result rank:", result.ndim)

    update_window_dims = list(update_window_dims)
    inserted_window_dims = list(inserted_window_dims)
    input_batching_dims = list(input_batching_dims)
    scatter_indices_batching_dims = list(scatter_indices_batching_dims)
    scatter_dims_to_operand_dims = list(scatter_dims_to_operand_dims)

    print("update_window_dims:", update_window_dims)
    print("inserted_window_dims:", inserted_window_dims)
    print("input_batching_dims:", input_batching_dims)
    print("scatter_indices_batching_dims:", scatter_indices_batching_dims)
    print("scatter_dims_to_operand_dims:", scatter_dims_to_operand_dims)
    print("index_vector_dim:", index_vector_dim)

    # For every update_index in index_space(updates)
    for update_index in np.ndindex(*updates.shape):
        #print("update_index:", update_index)
        # 1) compute update_scatter_dims = dims of update NOT in update_window_dims
        update_scatter_dims = [d for d in range(update_rank) if d not in update_window_dims]

        #print("update_scatter_dims:", update_scatter_dims)
        # build update_scatter_index tuple (order preserved)
        update_scatter_index = tuple(update_index[d] for d in update_scatter_dims)
        #print("update_scatter_index:", update_scatter_index)

        # 2) compute start_index from scatter_indices
        # If index_vector_dim < rank(scatter_indices), we build an index tuple with a slice
        if index_vector_dim < scatter_idx_rank:
            # build indexing tuple of length scatter_idx_rank:
            # at position index_vector_dim -> slice(None) (":")
            # else consume elements from update_scatter_index in order
            idx_tuple = []
            it = iter(update_scatter_index)
            for dim in range(scatter_idx_rank):
                if dim == index_vector_dim:
                    idx_tuple.append(slice(None))
                else:
                    # If update_scatter_index has fewer elements than needed, it will raise,
                    # but that indicates mismatched shapes/attributes.
                    idx_tuple.append(next(it))
            # evaluate
            start_vector = scatter_indices[tuple(idx_tuple)]
            # start_vector expected to be a 1-D array or scalar containing index vector.
            start_index = tuple(int(x) for x in np.array(start_vector).reshape(-1))
        else:
            # index_vector_dim >= rank(scatter_indices) -> index_vector is stored at the keyed location
            # The scatter_indices are indexed directly by update_scatter_index
            start_vector = scatter_indices[update_scatter_index]
            start_index = tuple(int(x) for x in np.array(start_vector).reshape(-1))
            
        print("start_index:", start_index)

        # 3) compute full_start_index over input axes
        full_start_index = [0] * input_rank
        # scatter_dims_to_operand_dims maps index-vector positions -> input dims
        for d_start, input_dim in enumerate(scatter_dims_to_operand_dims):
            # guard if start_index too short
            if d_start >= len(start_index):
                raise ValueError("start_index length mismatch vs scatter_dims_to_operand_dims")
            full_start_index[input_dim] = start_index[d_start]

        # 4) compute full_batching_index over input axes
        full_batching_index = [0] * input_rank
        # For every batching pair (i_batching), scatter_indices_batching_dims[i_batching] is a dim in scatter_indices
        # and input_batching_dims[i_batching] is the corresponding input dim.
        for i_batching, scatter_batch_dim in enumerate(scatter_indices_batching_dims):
            input_batch_dim = input_batching_dims[i_batching]
            # Determine the index inside update_scatter_index that corresponds to scatter_batch_dim.
            # Per the formal rule: index = d_start - (d_start < index_vector_dim ? 0 : 1)
            d_start = scatter_batch_dim
            if d_start < index_vector_dim:
                idx_in_update_scatter = d_start
            else:
                idx_in_update_scatter = d_start - 1
            # Guard
            if idx_in_update_scatter < 0 or idx_in_update_scatter >= len(update_scatter_index):
                raise ValueError("Batching mapping produced invalid index into update_scatter_index.")
            full_batching_index[input_batch_dim] = update_scatter_index[idx_in_update_scatter]

        # 5) compute update_window_index (tuple) and then full_window_index
        update_window_index = tuple(update_index[d] for d in update_window_dims)

        # full_window_index is built by iterating input axes; assign wi's into axes that are neither
        # inserted_window_dims nor input_batching_dims. The order of wi's is the order in update_window_dims.
        full_window_index = [0] * input_rank
        wi_iter = iter(update_window_index)
        for d_input in range(input_rank):
            if d_input in inserted_window_dims or d_input in input_batching_dims:
                # stays 0
                full_window_index[d_input] = 0
            else:
                # take next wi
                try:
                    full_window_index[d_input] = next(wi_iter)
                except StopIteration:
                    # If we run out of window indices, this likely indicates mismatched attributes/shapes.
                    full_window_index[d_input] = 0

        # 6) compute final result_index (elementwise sum)
        result_index = tuple(
            full_start_index[d] + full_batching_index[d] + full_window_index[d]
            for d in range(input_rank)
        )

        # 7) apply if in-bounds
        if in_bounds(result_index, inputs.shape):
            # convert update value if needed - here we just assume compatible types
            dest_value = result[result_index]
            upd_value = updates[update_index]
            new_value = update_computation(dest_value, upd_value)
            result[result_index] = new_value
        else:
            # out-of-bounds updates are ignored (per your description)
            pass

    return result


# --------------------------
# Example usage with the attributes and shapes from your example:
# input   : tensor<2x3x4x2xi64>
# indices : tensor<2x2x3x2xi64>
# updates : tensor<2x2x3x2x2xi64>
# Attributes from your sample:
#   update_window_dims = [3, 4]
#   inserted_window_dims = [1]
#   input_batching_dims = [0]
#   scatter_indices_batching_dims = [1]
#   scatter_dims_to_operand_dims = [2, 1]
#   index_vector_dim = 3
# Region performs add -> update_computation is addition
# --------------------------

def example_run():
    # shapes
    input_shape = (2, 3, 4, 2)
    indices_shape = (2, 2, 3, 2)
    updates_shape = (2, 2, 3, 2, 2)

    inputs = np.arange(np.prod(input_shape), dtype=np.int64).reshape(input_shape)
    scatter_indices = np.zeros(indices_shape, dtype=np.int64)
    for a, b, c in np.ndindex(indices_shape[0], indices_shape[1], indices_shape[2]):
        scatter_indices[a, b, c, 0] = (a + b + c) % input_shape[2]
        scatter_indices[a, b, c, 1] = (a + b + c) % input_shape[1]

    updates = np.ones(updates_shape, dtype=np.int64) * 10
    for idx in np.ndindex(*updates_shape):
        updates[idx] = (idx[-1] + idx[-2] + idx[0]) + 1

    attrs = dict(
        update_window_dims=[3, 4],
        inserted_window_dims=[1],
        input_batching_dims=[0],
        scatter_indices_batching_dims=[1],
        scatter_dims_to_operand_dims=[2, 1],
        index_vector_dim=3,
        indices_are_sorted=False,
        unique_indices=False
    )

    result = scatter_sim(
        inputs,
        scatter_indices,
        updates,
        update_window_dims=attrs['update_window_dims'],
        inserted_window_dims=attrs['inserted_window_dims'],
        input_batching_dims=attrs['input_batching_dims'],
        scatter_indices_batching_dims=attrs['scatter_indices_batching_dims'],
        scatter_dims_to_operand_dims=attrs['scatter_dims_to_operand_dims'],
        index_vector_dim=attrs['index_vector_dim'],
        indices_are_sorted=attrs['indices_are_sorted'],
        unique_indices=attrs['unique_indices'],
        update_computation=lambda a, b: a + b  # add region
    )

    print("inputs shape:", inputs.shape)
    print("scatter_indices shape:", scatter_indices.shape)
    print("updates shape:", updates.shape)
    print("result shape:", result.shape)
    # Show a few sample values for verification
    print("\nSample inputs[0,0,0,:] ->", inputs[0,0,0,:])
    print("Sample result[0,0,0,:] ->", result[0,0,0,:])

if __name__ == "__main__":
    example_run()
