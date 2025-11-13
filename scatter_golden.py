
def scatter(
  inputs: list,
  scatter_indices: list,
  updates: list,
  update_window_dims: list,
  inserted_window_dims: list,
  input_batching_dims: list,
  scatter_indices_batching_dims: list,
  scatter_dims_to_operand_dims: list,
  index_vector_dim: int,
  indices_are_sorted: bool,
  unique_indices: bool,
  update_computation = None,
):
  pass



if __name__ == "__main__":
    inputs = [
      [[1, 2], [3, 4], [5, 6], [7, 8]],
      [[9, 10], [11, 12], [13, 14], [15, 16]],
      [[17, 18], [19, 20], [21, 22], [23, 24]]]

    scatter_indices = [
      [[0, 2], [1, 0], [2, 1]],
      [[0, 1], [1, 0], [0, 9]]]

    updates = [
      [[[1, 1], [1, 1]],
       [[1, 1], [1, 1]],
       [[1, 1], [1, 1]]],
      [[[1, 1], [1, 1]],
       [[1, 1], [1, 1]],
       [[1, 1], [1, 1]]]]

    update_window_dims = [2, 3]
    inserted_window_dims = [0]
    input_batching_dims = []
    scatter_indices_batching_dims = []
    scatter_dims_to_operand_dims = [1, 0]
    index_vector_dim = 2
    indices_are_sorted = False
    unique_indices = False
    
    result = scatter(
      inputs,
      scatter_indices,
      updates,
      update_window_dims,
      inserted_window_dims,
      input_batching_dims,
      scatter_indices_batching_dims,
      scatter_dims_to_operand_dims,
      index_vector_dim,
      indices_are_sorted,
      unique_indices,
    )

    golden_result = [
      [[1, 2], [5, 6], [7, 8], [7, 8]],
      [[10, 11], [12, 13], [14, 15], [16, 17]],
      [[18, 19], [20, 21], [21, 22], [23, 24]]]





'''
I1)	inputs	variadic number of tensors or per-tensor quantized tensors	(C1), (C2), (C4-C6), (C11), (C13), (C18), (C21), (C23-C24)
(I2)	scatter_indices	tensor of integer type	(C4), (C15), (C19), (C22)
(I3)	updates	variadic number of tensors or per-tensor quantized tensors	(C3-C6), (C8)
(I4)	update_window_dims	1-dimensional tensor constant of type si64	(C2), (C4), (C7-C8)
(I5)	inserted_window_dims	1-dimensional tensor constant of type si64	(C2), (C4), (C9-C11)
(I6)	input_batching_dims	1-dimensional tensor constant of type si64	(C2), (C4), (C9), (C12-13), (C17-18), (C20)
(I7)	scatter_indices_batching_dims	1-dimensional tensor constant of type si64	(C14-C18)
(I8)	scatter_dims_to_operand_dims	1-dimensional tensor constant of type si64	(C19-C21)
(I9)	index_vector_dim	constant of type si64	(C4), (C16), (C19), (C22)
(I10)	indices_are_sorted	constant of type i1	
(I11)	unique_indices	constant of type i1	
(I12)	update_computation	function

Name	Type	Constraints
results	variadic number of tensors or per-tensor quantized tensors
'''

'''
%inputs = stablehlo.constant dense<[[[1, 2], [3, 4], [5, 6], [7, 8]],
                                      [[9, 10], [11, 12], [13, 14], [15, 16]],
                                      [[17, 18], [19, 20], [21, 22], [23, 24]]]> : tensor<3x4x2xi64>
  %scatter_indices = stablehlo.constant dense<[[[0, 2], [1, 0], [2, 1]],
                                               [[0, 1], [1, 0], [0, 9]]]> : tensor<2x3x2xi64>
  %updates = stablehlo.constant dense<1> : tensor<2x3x2x2xi64>

  scatter_dimension_numbers = #stablehlo.scatter<
      update_window_dims = [2, 3],
      inserted_window_dims = [0],
      scatter_dims_to_operand_dims = [1, 0],
      index_vector_dim = 2>,
    indices_are_sorted = false,
    unique_indices = false

    check.expect_eq_const %result, dense<[[[1, 2], [5, 6], [7, 8], [7, 8]],
                                        [[10, 11], [12, 13], [14, 15], [16, 17]],
                                        [[18, 19], [20, 21], [21, 22], [23, 24]]]> : tensor<3x4x2xi64>
'''