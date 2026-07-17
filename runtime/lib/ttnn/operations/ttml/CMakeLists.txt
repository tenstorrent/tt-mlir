# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

# Isolated wrapper library for ttml runtime ops.

if (NOT TTNN_RUNTIME_ENABLED)
  add_library(TTRuntimeTTMLOps INTERFACE)
  return()
endif()

add_library(TTRuntimeTTMLOps STATIC adamw.cpp)
set_property(TARGET TTRuntimeTTMLOps PROPERTY CXX_STANDARD 20)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|x86|i386|i686)$")
  target_compile_options(TTRuntimeTTMLOps PUBLIC -march=x86-64-v3)
endif()

# PUBLIC: op headers via operations/ttml/<op>.h + generated flatbuffer/runtime
# headers (matching the other runtime op libs). Only ttnn types are exposed.
target_include_directories(TTRuntimeTTMLOps PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTMLOps SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")

# PRIVATE: ttml source root is on this target's path only.
target_include_directories(TTRuntimeTTMLOps PRIVATE ${TTML_SOURCE_DIR})
target_link_libraries(TTRuntimeTTMLOps PRIVATE TTML_LIBRARY)
target_link_libraries(TTRuntimeTTMLOps PUBLIC TTNN_LIBRARY TTRuntimeTTNNTypes coverage_config)
add_dependencies(TTRuntimeTTMLOps TTML_LIBRARY tt-metal FBS_GENERATION RUNTIME_FBS_GENERATION)
