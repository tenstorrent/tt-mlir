find_package(MLIR REQUIRED CONFIG
             HINTS "")

set(TTMLIR_EXPORTED_TARGETS "MLIRTTDialect;MLIRTTNNDialect;TTMLIRTTNNUtils;MLIRTTKernelDialect;MLIRTTMetalDialect;MLIRTTNNTransforms;")
set(TTMLIR_CMAKE_DIR "/home/vwells/sources/tt-mlir/lib/cmake/ttmlir")
set(TTMLIR_BINARY_DIR "")
set(TTMLIR_INCLUDE_DIRS "/home/vwells/sources/tt-mlir/include;/home/vwells/sources/tt-mlir/include")
set(TTMLIR_LIBRARY_DIRS "")
set(TTMLIR_TOOLS_DIR "")

# Provide all our library targets to users.
if(NOT TARGET TTMLIRSupport)
  include("${TTMLIR_CMAKE_DIR}/TTMLIRTargets.cmake")
endif()
