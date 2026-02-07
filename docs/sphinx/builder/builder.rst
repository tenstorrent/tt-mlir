builder.base.builder
====================

This module provides the ``Builder`` base class and ``BuilderMeta`` metaclass
that form the foundation of the TTIR builder framework.  All dialect-specific
builders (``TTIRBuilder``, ``StableHLOBuilder``, ``TTNNBuilder``, etc.)
inherit from ``Builder``.

Builder Base Class
------------------

.. autoclass:: builder.base.builder.BuilderMeta
   :members:

.. autoclass:: builder.base.builder.Builder
   :members:
   :undoc-members:
   :show-inheritance:

Type Helpers
------------

.. autodata:: builder.base.builder_utils.Operand
.. autodata:: builder.base.builder_utils.Shape
.. autoclass:: builder.base.builder_utils.TypeInfo
   :members:

Decorator Utilities
-------------------

.. autofunction:: builder.base.builder_utils.tag
.. autofunction:: builder.base.builder_utils.parse
.. autofunction:: builder.base.builder_utils.split
