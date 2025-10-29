# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

find_package(nanobind CONFIG REQUIRED)

set(JIT_SOURCES
  lib/jit_cache.cpp
)

add_library(JITCPP SHARED ${JIT_SOURCES})
set_target_properties(JITCPP PROPERTIES CXX_STANDARD 20)

target_link_libraries(JITCPP
  PRIVATE
    MLIRIR
    MLIRSupport
    LLVMSupport
    TTMLIRCAPI
    TTMLIRCompilerStatic

    # libs for runtime + metal
    TTMLIRRuntime
    TTRuntimeTTNN
    TTRuntimeTTMetal
    TT_STL_LIBRARY

    # libs for fb translation
    TTNNTargetFlatbuffer
    TTKernelTargetCpp
    TTLLVMToDynamicLib
    MLIRLLVMToLLVMIRTranslation
)

target_include_directories(JITCPP
  SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>"
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

add_dependencies(JITCPP TTNN_LIBRARY tt-metal)

# Install JITCPP library to the same directory as the Python module
install(TARGETS JITCPP
  LIBRARY DESTINATION "${TTMLIR_PYTHON_PACKAGES_DIR}/ttnn_jit"
  COMPONENT SharedLib
)

nanobind_add_module(
  _ttnn_jit
  __init__.cpp
)
set_property(TARGET _ttnn_jit PROPERTY CXX_STANDARD 20)
set_target_properties(_ttnn_jit PROPERTIES
  INSTALL_RPATH "$ORIGIN"
  BUILD_RPATH "$ORIGIN;$<TARGET_FILE_DIR:JITCPP>"
  BUILD_WITH_INSTALL_RPATH FALSE
  LIBRARY_OUTPUT_DIRECTORY "${TTMLIR_PYTHON_PACKAGES_DIR}/ttnn_jit"
)
target_include_directories(_ttnn_jit
  SYSTEM PRIVATE ${pybind11_INCLUDE_DIRS}
  PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
)
target_link_libraries(_ttnn_jit
  PRIVATE
  JITCPP
  MLIRCAPIIR
  MLIRCAPITransforms
  TTMLIRCAPI
  TTMLIRCompilerStatic
)
