# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


if (NOT (TTMLIR_ENABLE_RUNTIME AND TT_RUNTIME_ENABLE_TTNN))
  add_library(ttml_metal_ops INTERFACE)
  return()
endif()

set(TTML_METAL_OP_SRCS
  ${TTML_SOURCE_DIR}/metal/optimizers/adamw/adamw.cpp
  ${TTML_SOURCE_DIR}/metal/optimizers/adamw/device/adamw_device_operation.cpp
  ${TTML_SOURCE_DIR}/metal/optimizers/adamw/device/adamw_program_factory.cpp
)
set_source_files_properties(${TTML_METAL_OP_SRCS} PROPERTIES GENERATED TRUE)

# STATIC on purpose. The archive tolerates any ttnn/ttml::core symbols left
# undefined by ops we don't dispatch; the runtime only references the ops wired
# in ttml.cpp, so only those objects get pulled into libTTMLIRRuntime.so and
# must fully resolve. It also keeps this out of libTTMLIRRuntime.so's NEEDED
# list (no ttrt ImportError for a satellite .so).
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
