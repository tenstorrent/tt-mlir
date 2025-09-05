set(ttmlir_cmake_builddir "${CMAKE_BINARY_DIR}/lib/cmake/ttmlir")

set_property(GLOBAL APPEND PROPERTY TTMLIR_EXPORTS "TTMLIRTTKernelToEmitC;MLIRTTTransforms;MLIRTTCoreDialect;MLIRTTNNDialect;TTMLIRTTNNUtils;MLIRTTKernelDialect;MLIRTTMetalDialect;TTNNOpModelLib;MLIRTTNNInterfaces;coverage_config")
get_property(TTMLIR_EXPORTS GLOBAL PROPERTY TTMLIR_EXPORTS)
export(TARGETS ${TTMLIR_EXPORTS} FILE ${ttmlir_cmake_builddir}/TTMLIRTargets.cmake)

set(TTMLIR_CONFIG_CMAKE_DIR "${ttmlir_cmake_builddir}")
set(TTMLIR_CONFIG_LIBRARY_DIRS "${TTMLIR_LIBRARY_DIR}")
set(TTMLIR_CONFIG_INCLUDE_EXPORTS "include(\"\${TTMLIR_CMAKE_DIR}/TTMLIRTargets.cmake\")")
set(TTMLIR_CONFIG_INCLUDE_DIRS
  "${TTMLIR_BINARY_DIR}/include"
  "${TTMLIR_SOURCE_DIR}/include"
  )

configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/TTMLIRConfig.cmake.in
  ${ttmlir_cmake_builddir}/TTMLIRConfig.cmake
  @ONLY)
set(TTMLIR_CONFIG_CMAKE_DIR)
set(TTMLIR_CONFIG_LIBRARY_DIRS)
set(TTMLIR_CONFIG_INCLUDE_EXPORTS)
set(TTMLIR_CONFIG_INCLUDE_DIRS)
