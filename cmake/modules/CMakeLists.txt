set(ttmlir_cmake_builddir "${CMAKE_BINARY_DIR}/lib/cmake/ttmlir")
set(ttmlir_cmake_installdir "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/ttmlir")

set_property(GLOBAL APPEND PROPERTY TTMLIR_EXPORTS "TTMLIRTTKernelToEmitC;MLIRTTTransforms;MLIRTTCoreDialect;MLIRTTNNDialect;TTMLIRTTNNUtils;MLIRTTKernelDialect;MLIRTTMetalDialect;TTNNOpModelLib;MLIRTTNNInterfaces;coverage_config")
get_property(TTMLIR_EXPORTS GLOBAL PROPERTY TTMLIR_EXPORTS)
export(TARGETS ${TTMLIR_EXPORTS} FILE ${ttmlir_cmake_builddir}/TTMLIRTargets.cmake)

# Set library and tools directories, defaulting to standard locations if not set.
set(ttmlir_library_dir "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}")
if(NOT ttmlir_library_dir)
  set(ttmlir_library_dir "${CMAKE_BINARY_DIR}/lib")
endif()

set(ttmlir_tools_dir "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}")
if(NOT ttmlir_tools_dir)
  set(ttmlir_tools_dir "${CMAKE_BINARY_DIR}/bin")
endif()

set(TTMLIR_CONFIG_CMAKE_DIR "${ttmlir_cmake_builddir}")
set(TTMLIR_CONFIG_BINARY_DIR "${TTMLIR_BINARY_DIR}")
set(TTMLIR_CONFIG_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")
set(TTMLIR_CONFIG_LIBRARY_DIRS "${ttmlir_library_dir}")
set(TTMLIR_CONFIG_TOOLS_DIR "${ttmlir_tools_dir}")
set(TTMLIR_CONFIG_MLIR_CMAKE_DIR "${MLIR_DIR}")
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
set(TTMLIR_CONFIG_BINARY_DIR)
set(TTMLIR_CONFIG_INSTALL_PREFIX)
set(TTMLIR_CONFIG_LIBRARY_DIRS)
set(TTMLIR_CONFIG_TOOLS_DIR)
set(TTMLIR_CONFIG_MLIR_CMAKE_DIR)
set(TTMLIR_CONFIG_INCLUDE_EXPORTS)
set(TTMLIR_CONFIG_INCLUDE_DIRS)

# Install the config files and targets.
# Use a unified export name for all TTMLIR targets.
set(TTMLIR_EXPORT_NAME "TTMLIRTargets")

install(FILES ${ttmlir_cmake_builddir}/TTMLIRConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ttmlir
  COMPONENT SharedLib)

install(EXPORT ${TTMLIR_EXPORT_NAME}
  FILE TTMLIRTargets.cmake
  NAMESPACE TTMLIR::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/ttmlir
  COMPONENT SharedLib)

# Install exported targets with the unified export name.
# Note: Targets may also be installed with other export names elsewhere.
install(TARGETS ${TTMLIR_EXPORTS}
  EXPORT ${TTMLIR_EXPORT_NAME}
  COMPONENT SharedLib)
