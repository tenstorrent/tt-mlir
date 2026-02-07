builder.ttir.ttir_builder
=========================

This module provides the ``TTIRBuilder`` class which emits TTIR
(Tenstorrent Intermediate Representation) dialect operations.  It extends
:class:`~builder.base.builder.Builder` with methods for every supported
TTIR op—including collective communication, data-movement, elementwise,
reduction, and neural-network operations.

TTIRBuilder
-----------

.. autoclass:: builder.ttir.ttir_builder.TTIRBuilder
   :members:
   :undoc-members:
   :show-inheritance:
   :inherited-members:
