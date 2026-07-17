# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0


if (NOT TTNN_RUNTIME_ENABLED)
  add_library(TTRuntimeTTMLOps INTERFACE)
  return()
endif()

set(TTML_OP_SRCS
  ttml.cpp
  adamw.cpp
)

add_library(TTRuntimeTTMLOps SHARED ${TTML_OP_SRCS})
set_target_properties(TTRuntimeTTMLOps PROPERTIES
  OUTPUT_NAME TTRuntimeTTML
  CXX_STANDARD 20
  CXX_VISIBILITY_PRESET hidden
  VISIBILITY_INLINES_HIDDEN ON
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/runtime/lib
  INSTALL_RPATH "$ORIGIN;${TTMETAL_LIBRARY_DIR}"
  BUILD_RPATH "$ORIGIN;${TTMETAL_LIBRARY_DIR}"
)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|x86|i386|i686)$")
  target_compile_options(TTRuntimeTTMLOps PRIVATE -march=x86-64-v3)
endif()

# PUBLIC: op header via operations/ttml/<op>.h + generated flatbuffer/runtime
# headers. Only ttnn types are exposed on the run() boundary.
target_include_directories(TTRuntimeTTMLOps PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTMLOps SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
# ttml source root only on this target.
target_include_directories(TTRuntimeTTMLOps PRIVATE ${TTML_SOURCE_DIR})

# The target is built with hidden visibility (above); only the op entry points
# annotated with TTNN_RUNTIME_TTML_EXPORT are exported. Generate that macro into
# the shared runtime include dir so both this library (adamw.cpp) and its in-tree
# caller (program_executor.cpp, in TTRuntimeTTNN) resolve the same header.
include(GenerateExportHeader)
generate_export_header(TTRuntimeTTMLOps
  BASE_NAME TTNN_RUNTIME_TTML
  EXPORT_MACRO_NAME TTNN_RUNTIME_TTML_EXPORT
  EXPORT_FILE_NAME ${PROJECT_BINARY_DIR}/runtime/include/ttml_runtime_export.h
)

target_link_libraries(TTRuntimeTTMLOps PRIVATE TTML_LIBRARY)
target_link_libraries(TTRuntimeTTMLOps PUBLIC ${METAL_LIBS} coverage_config)

# libttml.a is a prebuilt archive, so CXX_VISIBILITY_PRESET can't reach it; hide
# its symbols with --exclude-libs so tt-train's ttnn/flatbuffers instantiations
# stay internal and never merge with the canonical process-wide copies -- the
# thing that unbreaks nanobind type identity for consumers that also load this
# .so (ttnn-jit).
target_link_options(TTRuntimeTTMLOps PRIVATE "LINKER:--exclude-libs,libttml.a")

# Satellite of libTTMLIRRuntime.so: opt out of the global `-z defs` so the
# runtime/spdlog symbols left undefined above are permitted and bound at load
# time (the trailing `-z undefs` overrides the global `-z defs`).
target_link_options(TTRuntimeTTMLOps PRIVATE "LINKER:-z,undefs")

add_dependencies(TTRuntimeTTMLOps TTML_LIBRARY tt-metal FBS_GENERATION RUNTIME_FBS_GENERATION)

# Install alongside libTTMLIRRuntime.so (same component, so consumers that ship
# the runtime .so -- the ttrt wheel, tt-xla -- get this beside it).
install(TARGETS TTRuntimeTTMLOps
  LIBRARY
    DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
    COMPONENT SharedLib
)
