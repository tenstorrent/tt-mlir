ttir\_builder.ttir\_golden
===========================

.. autofunction:: ttir_builder.ttir_golden.cbrt_golden

.. autofunction:: ttir_builder.ttir_golden.conv2d_golden

.. autofunction:: ttir_builder.ttir_golden.conv_transpose2d_golden

.. autofunction:: ttir_builder.ttir_golden.linear_golden

.. autofunction:: ttir_builder.ttir_golden.max_pool2d_golden

.. autofunction:: ttir_builder.ttir_golden.pad_golden

.. autofunction:: ttir_builder.ttir_golden.select_golden

.. autofunction:: ttir_builder.ttir_golden.index_golden

.. autofunction:: ttir_builder.ttir_golden.slice_golden

.. autofunction:: ttir_builder.ttir_golden.gather_golden

.. autofunction:: ttir_builder.ttir_golden.embedding_golden

.. autofunction:: ttir_builder.ttir_golden.upsample2d_golden

.. autofunction:: ttir_builder.ttir_golden.arange_golden

.. autofunction:: ttir_builder.ttir_golden.quantize_golden

.. autofunction:: ttir_builder.ttir_golden.requantize_golden

.. autofunction:: ttir_builder.ttir_golden.argmax_golden

.. autofunction:: ttir_builder.ttir_golden.dot_general_golden

.. autofunction:: ttir_builder.ttir_golden.tilize_golden

.. autofunction:: ttir_builder.ttir_golden.untilize_golden

.. autofunction:: ttir_builder.ttir_golden.fill_cache_golden

.. autofunction:: ttir_builder.ttir_golden.update_cache_golden

.. autofunction:: ttir_builder.ttir_golden.mesh_shard_golden

.. autofunction:: ttir_builder.ttir_golden.all_gather_golden

.. autofunction:: ttir_builder.ttir_golden.all_reduce_golden

.. autofunction:: ttir_builder.ttir_golden.reduce_scatter_golden

.. autofunction:: ttir_builder.ttir_golden.collective_permute_golden

.. autofunction:: ttir_builder.ttir_golden.all_to_all_golden

.. autofunction:: ttir_builder.ttir_golden.collective_broadcast_golden

.. autofunction:: ttir_builder.ttir_golden.max_golden

.. autofunction:: ttir_builder.ttir_golden.prod_golden

.. autofunction:: ttir_builder.ttir_golden.get_golden_function

GOLDEN_MAPPINGS Dictionary
--------------------------

.. data:: ttir_builder.ttir_golden.GOLDEN_MAPPINGS

   Dictionary mapping TTIR operation classes to their corresponding golden functions.

   This dictionary provides a centralized mapping between TTIR operation types and their
   PyTorch-based golden reference implementations. Each key is a TTIR operation class
   (e.g., ``ttir.AbsOp``) and each value is the corresponding golden function that computes
   the expected output for that operation.

   The mapping supports:
       - Elementwise unary operations (abs, ceil, cos, etc.)
       - Elementwise binary operations (add, multiply, subtract, etc.)
       - Elementwise ternary operations (where, select, etc.)
       - Comparison operations (eq, ne, lt, gt, etc.)
       - Bitwise operations (and, or, xor, not)
       - Reduction operations (sum, mean, max, min, etc.)
       - Tensor manipulation (transpose, concat, reshape, etc.)
       - Neural network operations (matmul, embedding, conv2d, etc.)
       - Layout operations (to_layout, view_layout)
       - Quantization operations (quantize, dequantize, requantize)
       - Collective communication operations (all_gather, all_reduce, etc.)

   **Example usage:**

   .. code-block:: python

       golden_fn = GOLDEN_MAPPINGS.get(ttir.AbsOp)
       if golden_fn:
           result = golden_fn(input_tensor)
