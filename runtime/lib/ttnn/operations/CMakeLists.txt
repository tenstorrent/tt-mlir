set(TTNN_OPS_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/include/tt/runtime/ttnn/operations/utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/conv/conv2d/conv2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/creation/empty/empty.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/creation/full/full.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/concat/concat.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/reshape/reshape.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/data_movement/transpose/transpose.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/deletion/dealloc/dealloc.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise/binary/binary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/eltwise/unary/unary.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/embedding/embedding/embedding.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/to_device/to_device.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/to_layout/to_layout.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/layout/to_memory_config/to_memory_config.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/matmul/matmul/matmul.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/normalization/softmax/softmax.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/pool/maxpool/maxpool2d.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reduction/reduction/reduction.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/context/get_device/get_device.cpp
)

add_library(TTRuntimeTTNNOps
  STATIC
  ${TTNN_OPS_SRCS}
)

set_property(TARGET TTRuntimeTTNNOps PROPERTY CXX_STANDARD 20)
target_compile_options(TTRuntimeTTNNOps PUBLIC -mavx -mavx2)
target_include_directories(TTRuntimeTTNNOps PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/operations/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNOps PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_link_libraries(TTRuntimeTTNNOps PUBLIC TTNN_LIBRARY)
add_dependencies(TTRuntimeTTNNOps TTNN_LIBRARY tt-metal FBS_GENERATION)
