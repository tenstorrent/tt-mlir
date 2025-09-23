include(BuildFlatbuffers)

set(TTNN_OPERATIONS_FBS_GEN_SOURCES
  operations/ccl.fbs
  operations/get_device.fbs
  operations/configs.fbs
  operations/conv.fbs
  operations/cpu.fbs
  operations/creation.fbs
  operations/data_movement.fbs
  operations/deallocate.fbs
  operations/eltwise.fbs
  operations/embedding.fbs
  operations/generic_op.fbs
  operations/kv_cache.fbs
  operations/layout.fbs
  operations/load_cached.fbs
  # ANCHOR: adding_an_op_matmul_fbs_cmake
  operations/matmul.fbs
  # ANCHOR_END: adding_an_op_matmul_fbs_cmake
  operations/mlir_native.fbs
  operations/moreh_cumsum.fbs
  operations/normalization.fbs
  operations/pool.fbs
  operations/rand.fbs
  operations/reduction.fbs
  operations/tensor_serialization.fbs
  operations/trace.fbs
  operations/transformer.fbs
)

set(TTNN_FBS_GEN_SOURCES
  types.fbs
  ${TTNN_OPERATIONS_FBS_GEN_SOURCES}
  program.fbs
  binary.fbs
)

build_flatbuffers("ttnn" "${TTNN_FBS_GEN_SOURCES}" TTNN_FBS COMMON_FBS)
