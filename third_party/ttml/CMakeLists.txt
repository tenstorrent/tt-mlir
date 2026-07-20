# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

if (NOT (TTMLIR_ENABLE_RUNTIME AND TT_RUNTIME_ENABLE_TTNN))
  add_library(ttml_metal_ops INTERFACE)
  return()
endif()

file(GLOB_RECURSE TTML_METAL_OP_SRCS CONFIGURE_DEPENDS
  ${TTML_SOURCE_DIR}/metal/ops/*.cpp
  ${TTML_SOURCE_DIR}/metal/optimizers/*.cpp
)

list(FILTER TTML_METAL_OP_SRCS EXCLUDE REGEX "/kernels/")
list(FILTER TTML_METAL_OP_SRCS EXCLUDE REGEX "/layernorm_bw/")

if (NOT TTML_METAL_OP_SRCS)
  message(FATAL_ERROR
    "No ttml metal op sources under ${TTML_SOURCE_DIR}/metal. The tt-metal "
    "checkout must exist at configure time (it is fetched at build time for a "
    "clean ExternalProject build). Pre-fetch tt-metal or set "
    "-DTTMLIR_TTMETAL_SOURCE_DIR=<checkout>.")
endif()

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
  ${TTNN_LIBRARY_PATH}
  ${TTMETAL_LIBRARY_PATH}
  ${TT_STL_LIBRARY_PATH}
  ${DEVICE_LIBRARY_PATH}
  ${TRACY_LIBRARY_PATH}
  coverage_config
)

add_dependencies(ttml_metal_ops tt-metal)
