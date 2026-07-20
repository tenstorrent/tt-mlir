# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


if (NOT TTNN_RUNTIME_ENABLED)
  add_library(TTRuntimeTTMLOps INTERFACE)
  return()
endif()

add_library(TTRuntimeTTMLOps STATIC
  ttml.cpp
  adamw.cpp
)
set_property(TARGET TTRuntimeTTMLOps PROPERTY CXX_STANDARD 20)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|x86|i386|i686)$")
  target_compile_options(TTRuntimeTTMLOps PRIVATE -march=x86-64-v3)
endif()

target_include_directories(TTRuntimeTTMLOps PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTMLOps SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
target_include_directories(TTRuntimeTTMLOps PRIVATE ${TTML_SOURCE_DIR})

target_link_libraries(TTRuntimeTTMLOps PUBLIC ttml_metal_ops ${METAL_LIBS} coverage_config)

add_dependencies(TTRuntimeTTMLOps tt-metal FBS_GENERATION RUNTIME_FBS_GENERATION)
