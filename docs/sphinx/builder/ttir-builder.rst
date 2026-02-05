TTIR Builder Module
===================

The ``builder.ttir.ttir_builder`` module provides the primary interface for 
constructing TTIR (Tenstorrent IR) operations programmatically. It extends the 
base builder infrastructure with TTIR-specific operation builders.

Overview
--------

TTIRBuilder enables:

- **Operation Construction**: Methods for creating all supported TTIR operations
- **Type Inference**: Automatic output type computation based on input operands
- **Golden Value Management**: Integration with test infrastructure for verification
- **Function Building**: Decorators and utilities for constructing MLIR functions

Usage Example
-------------

.. code-block:: python

    from builder.ttir.ttir_builder import TTIRBuilder
    from builder.base.builder_apis import compile_and_execute_ttir

    def my_module(builder: TTIRBuilder):
        @builder.func([(32, 32)], [torch.float32])
        def my_function(in0, builder):
            # Apply ReLU activation
            result = builder.relu(in0)
            return result

    compile_and_execute_ttir(my_module, target="ttnn", device=device)

TTIRBuilder Class
-----------------

.. autoclass:: builder.ttir.ttir_builder.TTIRBuilder
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:

Operation Categories
--------------------

The TTIRBuilder provides methods for various operation categories:

**Elementwise Operations**
    ``relu``, ``sigmoid``, ``exp``, ``log``, ``sqrt``, ``abs``, ``neg``, etc.

**Reduction Operations**
    ``sum``, ``mean``, ``max``, ``prod`` with configurable reduction dimensions

**Matrix Operations**
    ``matmul``, ``dot_general``, ``linear`` for tensor contractions

**Shape Operations**
    ``reshape``, ``transpose``, ``concat``, ``slice``, ``broadcast``

**Comparison Operations**
    ``eq``, ``ne``, ``lt``, ``le``, ``gt``, ``ge`` for element-wise comparisons

See Also
--------

- :doc:`builder` - Core builder infrastructure
- :doc:`builder-utils` - Utility functions and helpers
