builder.base.builder\_utils
===========================

Shared utility functions for building, compiling, and executing MLIR modules.

Module Building
---------------

.. autofunction:: builder.base.builder_utils.build_ttir_module

.. autofunction:: builder.base.builder_utils.compile_ttir_to_flatbuffer

.. autofunction:: builder.base.builder_utils.build_stablehlo_module

Pipeline Utilities
------------------

.. autofunction:: builder.base.builder_utils.run_ttir_pipeline

.. autofunction:: builder.base.builder_utils.create_custom_ttir_pipeline_fn

Tensor Layout
-------------

.. autofunction:: builder.base.builder_utils.get_metal_tensor_layout

.. autofunction:: builder.base.builder_utils.get_target_path

.. autofunction:: builder.base.builder_utils.get_artifact_dir

Helper Functions
----------------

.. autofunction:: builder.base.builder_utils.process_multi_return_result
