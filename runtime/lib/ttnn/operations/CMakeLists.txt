if (NOT TTNN_RUNTIME_ENABLED)
  add_library(TTRuntimeTTNNOps INTERFACE)
  return()
endif()

set(TTNN_OPS_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/cache/load_cached.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ccl/all_gather.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ccl/collective_permute.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ccl/mesh_shard.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ccl/point_to_point.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ccl/reduce_scatter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/context/get_device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv/conv2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv/conv_transpose2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv/prepare_conv2d_weights.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv/prepare_conv2d_bias.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/cpu/cpu.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/creation/arange.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/creation/constant.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/creation/empty.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/creation/full.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/creation/full_with.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/concat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/pad.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/permute.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/repeat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/repeat_interleave.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/reshape.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/scatter.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/slice.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/sort.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/transpose.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/write_tensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/deletion/deallocate.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise/binary/binary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise/binary/binary_composite.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise/quantization/quantization.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise/ternary/where.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise/unary/unary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise/unary/unary_composite.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/embedding/embedding.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/embedding/embedding_backward.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/generic/generic_op.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kv_cache/fill_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/kv_cache/update_cache.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/from_device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/to_device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/to_dtype.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/to_layout.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/to_memory_config.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/typecast.cpp
  # ANCHOR: adding_an_op_matmul_runtime_cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul/matmul.cpp
  # ANCHOR_END: adding_an_op_matmul_runtime_cmake
  ${CMAKE_CURRENT_SOURCE_DIR}/mlir_native/func_call.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/moreh/moreh_cumsum.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/batch_norm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/rms_norm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/softmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pool/pool2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pool/upsample.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/rand/rand.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduction/argmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduction/prod.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduction/reduction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tensor_serialization/dump_tensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tensor_serialization/load_tensor.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/trace/begin_trace_capture.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/trace/capture_or_execute_trace.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/trace/end_trace_capture.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/trace/execute_trace.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transformer/concatenate_heads.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transformer/rotary_embedding_llama.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transformer/nlp_create_qkv_heads_decode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transformer/nlp_concat_heads.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transformer/nlp_concat_heads_decode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transformer/scaled_dot_product_attention.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/transformer/scaled_dot_product_attention_decode.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/utils/utils.cpp
)

add_library(TTRuntimeTTNNOps
  STATIC
  ${TTNN_OPS_SRCS}
)

set_property(TARGET TTRuntimeTTNNOps PROPERTY CXX_STANDARD 20)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|x86|i386|i686)$")
  target_compile_options(TTRuntimeTTNNOps PUBLIC -march=x86-64-v3)
endif()
target_include_directories(TTRuntimeTTNNOps PRIVATE
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn
)
target_include_directories(TTRuntimeTTNNOps PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)

target_include_directories(TTRuntimeTTNNOps SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(TTRuntimeTTNNOps PUBLIC TTNN_LIBRARY TTRuntimeTTNNDebug TTRuntimeTTNNTypes TTRuntimeTTNNUtils TRACY_LIBRARY)
target_link_libraries(TTRuntimeTTNNOps PUBLIC coverage_config)


add_dependencies(TTRuntimeTTNNOps TTNN_LIBRARY tt-metal FBS_GENERATION TTRuntimeTTNNTypes TTRuntimeTTNNDebug)
