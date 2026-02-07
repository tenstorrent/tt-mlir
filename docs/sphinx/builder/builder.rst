Builder Module
==============

The ``builder.base.builder`` module provides core infrastructure for constructing 
and managing MLIR-based computational graphs. It implements a flexible builder 
pattern that allows programmatic construction of tensor operations.

Overview
--------

The Builder module provides:

- **Type Management**: Classes for representing tensor types, shapes, and data types
- **Golden Value Tracking**: Infrastructure for storing expected output values for verification
- **Operand Management**: Type aliases and utilities for working with MLIR SSA values

Key Classes
-----------

TypeInfo
~~~~~~~~

Represents complete type information for tensor operands including shape, element type,
and optional encoding information.

.. autoclass:: builder.base.builder.TypeInfo
   :members:
   :undoc-members:
   :show-inheritance:

Golden
~~~~~~

Container for golden (expected) values used in testing and verification.

.. autoclass:: builder.base.builder.Golden
   :members:
   :undoc-members:

GoldenCheckLevel
~~~~~~~~~~~~~~~~

Enumeration defining verification stringency levels for golden comparisons.

.. autoclass:: builder.base.builder.GoldenCheckLevel
   :members:
   :undoc-members:

Type Aliases
------------

.. autodata:: builder.base.builder.Operand
   :annotation: = Union[OpResult, BlockArgument]

   Represents an MLIR SSA value that can be used as an operand to operations.

.. autodata:: builder.base.builder.Shape
   :annotation: = Tuple[int, ...]

   Represents the shape of a tensor as a tuple of dimension sizes.

See Also
--------

- :doc:`ttir-builder` - TTIR-specific builder extensions
- :doc:`builder-utils` - Utility functions for builder operations
