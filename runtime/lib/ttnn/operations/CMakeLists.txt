set(TTNN_OPS_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tt/runtime/ttnn/operations/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tt/runtime/ttnn/operations/eltwise/binary/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tt/runtime/ttnn/operations/eltwise/unary/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tt/runtime/ttnn/operations/eltwise/ternary/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ccl/all_gather.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ccl/reduce_scatter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ccl/mesh_shard.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv/conv2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/creation/arange.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/creation/constant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv/conv_transpose2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/creation/empty.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/creation/zeros.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/creation/ones.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/creation/full.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/concat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/permute.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/reshape.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/repeat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/repeat_interleave.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/slice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/transpose.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/pad.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/deletion/deallocate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise/binary/binary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise/binary/binary_composite.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise/unary/unary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise/unary/unary_composite.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise/ternary/ternary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/embedding/embedding.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kv_cache/fill_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kv_cache/update_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/embedding/embedding_backward.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/to_device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/from_device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/to_layout.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/to_dtype.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/typecast.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/to_memory_config.cpp
  # ANCHOR: adding_an_op_matmul_runtime_cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul/matmul.cpp
  # ANCHOR_END: adding_an_op_matmul_runtime_cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/moreh/moreh_cumsum.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/softmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pool/maxpool2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pool/upsample.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduction/argmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduction/prod.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduction/reduction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/context/get_device.cpp
)

add_library(TTRuntimeTTNNOps
  STATIC
  ${TTNN_OPS_SRCS}
)

set_property(TARGET TTRuntimeTTNNOps PROPERTY CXX_STANDARD 20)
target_compile_options(TTRuntimeTTNNOps PUBLIC -mavx -mavx2)
target_include_directories(TTRuntimeTTNNOps PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn
)
target_include_directories(TTRuntimeTTNNOps PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/operations/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)

target_include_directories(TTRuntimeTTNNOps SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(TTRuntimeTTNNOps PUBLIC TTNN_LIBRARY TTRuntimeTTNNTypes TTRuntimeTTNNUtils)

if (TT_RUNTIME_ENABLE_PERF_TRACE)
  target_link_libraries(TTRuntimeTTNNOps PUBLIC TRACY_LIBRARY)
endif()


add_dependencies(TTRuntimeTTNNOps TTNN_LIBRARY tt-metal FBS_GENERATION TTRuntimeTTNNTypes TTRuntimeTTNNUtils)
