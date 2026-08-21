# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


# Built for the TTNN runtime (which dispatches these ops on device) and for
# OpModel (which queries their constraints/runtime through ttnn::graph). Either
# consumer alone is enough; both gate on tt-metal, which third_party/CMakeLists
# builds under the same `RUNTIME OR OPMODEL` condition.
if (NOT ((TTMLIR_ENABLE_RUNTIME AND TT_RUNTIME_ENABLE_TTNN) OR TTMLIR_ENABLE_OPMODEL))
  add_library(ttml_metal_ops INTERFACE)
  return()
endif()

set(TTML_METAL_OP_SRCS
  ${TTML_SOURCE_DIR}/metal/optimizers/adamw/adamw.cpp
  ${TTML_SOURCE_DIR}/metal/optimizers/adamw/device/adamw_device_operation.cpp
  ${TTML_SOURCE_DIR}/metal/optimizers/adamw/device/adamw_program_factory.cpp
  ${TTML_SOURCE_DIR}/metal/ops/sdpa_fw/sdpa_fw.cpp
  ${TTML_SOURCE_DIR}/metal/ops/sdpa_fw/device/sdpa_fw_device_operation.cpp
  ${TTML_SOURCE_DIR}/metal/ops/sdpa_fw/device/sdpa_fw_program_factory.cpp
  ${TTML_SOURCE_DIR}/metal/ops/sdpa_bw/sdpa_bw.cpp
  ${TTML_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_q_device_operation.cpp
  ${TTML_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_q_program_factory.cpp
  ${TTML_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_kv_device_operation.cpp
  ${TTML_SOURCE_DIR}/metal/ops/sdpa_bw/device/sdpa_bw_kv_program_factory.cpp
  ${TTML_SOURCE_DIR}/metal/ops/cross_entropy_fw/cross_entropy_fw.cpp
  ${TTML_SOURCE_DIR}/metal/ops/cross_entropy_fw/device/cross_entropy_fw_device_operation.cpp
  ${TTML_SOURCE_DIR}/metal/ops/cross_entropy_fw/device/cross_entropy_fw_program_factory.cpp
)
set_source_files_properties(${TTML_METAL_OP_SRCS} PROPERTIES GENERATED TRUE)

add_library(ttml_metal_ops STATIC ${TTML_METAL_OP_SRCS})
set_property(TARGET ttml_metal_ops PROPERTY CXX_STANDARD 20)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|x86|i386|i686)$")
  target_compile_options(ttml_metal_ops PRIVATE -march=x86-64-v3)
endif()

set_source_files_properties(${TTML_METAL_OP_SRCS} PROPERTIES
  COMPILE_OPTIONS "-Wno-error;-Wno-missing-designated-field-initializers;-Wno-c++11-narrowing")

target_include_directories(ttml_metal_ops PRIVATE ${TTML_SOURCE_DIR})
target_include_directories(ttml_metal_ops SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

target_link_libraries(ttml_metal_ops PUBLIC
  TTNN_LIBRARY
  TTMETAL_LIBRARY
  TT_STL_LIBRARY
  DEVICE_LIBRARY
  TRACY_LIBRARY
  coverage_config
)

add_dependencies(ttml_metal_ops tt-metal)
