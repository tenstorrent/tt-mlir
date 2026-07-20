# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


if (NOT TTNN_RUNTIME_ENABLED)
  add_library(TTRuntimeTTMLOps INTERFACE)
  return()
endif()

set(TTML_ENABLED_OP_DIRS
  optimizers/adamw
)

set(TTML_METAL_OP_SRCS "")
foreach(_op_dir ${TTML_ENABLED_OP_DIRS})
  file(GLOB_RECURSE _op_srcs CONFIGURE_DEPENDS
    ${TTML_SOURCE_DIR}/metal/${_op_dir}/*.cpp)
  list(FILTER _op_srcs EXCLUDE REGEX "/kernels/")
  list(APPEND TTML_METAL_OP_SRCS ${_op_srcs})
endforeach()

add_library(TTRuntimeTTMLOps STATIC
  ttml.cpp
  adamw.cpp
  ${TTML_METAL_OP_SRCS}
)
set_property(TARGET TTRuntimeTTMLOps PROPERTY CXX_STANDARD 20)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|x86|i386|i686)$")
  target_compile_options(TTRuntimeTTMLOps PRIVATE -march=x86-64-v3)
endif()

if(TTML_METAL_OP_SRCS)
  set_source_files_properties(${TTML_METAL_OP_SRCS} PROPERTIES
    COMPILE_OPTIONS "-Wno-error;-Wno-missing-designated-field-initializers")
endif()

target_include_directories(TTRuntimeTTMLOps PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTMLOps SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
# tt-train source root: metal/common headers, metal/ttnn_all_includes.hpp, and
# each op's own device/*.hpp.
target_include_directories(TTRuntimeTTMLOps PRIVATE ${TTML_SOURCE_DIR})

target_link_libraries(TTRuntimeTTMLOps PUBLIC ${METAL_LIBS} coverage_config)

add_dependencies(TTRuntimeTTMLOps tt-metal FBS_GENERATION RUNTIME_FBS_GENERATION)
