# InstallComponents.cmake
# Handles installation of required TT-NN components for cpp standalone builds

# Print one message per install command instead of per file
set(CMAKE_INSTALL_MESSAGE LAZY)

# Install directory for TT-NN components
if(NOT DEFINED TTNN_INSTALL_DIR)
  set(TTNN_INSTALL_DIR "${CMAKE_CURRENT_BINARY_DIR}/../ttnn-install" CACHE PATH "Directory to install TT-NN components")
endif()

# Path to TT-NN build directory
if(NOT DEFINED TT_METAL_BUILD_HOME)
  if(DEFINED ENV{TT_METAL_BUILD_HOME})
    set(TT_METAL_BUILD_HOME "$ENV{TT_METAL_BUILD_HOME}")
  elseif(DEFINED ENV{TT_METAL_HOME})
    set(TT_METAL_BUILD_HOME "$ENV{TT_METAL_HOME}/build")
  else()
    message(FATAL_ERROR "TT_METAL_BUILD_HOME or TT_METAL_HOME environment variable must be set")
  endif()
endif()

# Create install directory
add_custom_command(
  OUTPUT ${TTNN_INSTALL_DIR}
  COMMAND ${CMAKE_COMMAND} -E make_directory ${TTNN_INSTALL_DIR}
  COMMENT "Creating TT-NN install directory"
)

# Install metalium-runtime component
add_custom_command(
  OUTPUT ${TTNN_INSTALL_DIR}/.metalium-runtime-installed
  DEPENDS ${TTNN_INSTALL_DIR}
  COMMAND cmake --install ${TT_METAL_BUILD_HOME} --prefix ${TTNN_INSTALL_DIR} --component metalium-runtime
  COMMAND ${CMAKE_COMMAND} -E touch ${TTNN_INSTALL_DIR}/.metalium-runtime-installed
  COMMENT "Installing metalium-runtime component"
)

# Install metalium-dev component
add_custom_command(
  OUTPUT ${TTNN_INSTALL_DIR}/.metalium-dev-installed
  DEPENDS ${TTNN_INSTALL_DIR}/.metalium-runtime-installed
  COMMAND cmake --install ${TT_METAL_BUILD_HOME} --prefix ${TTNN_INSTALL_DIR} --component metalium-dev
  COMMAND ${CMAKE_COMMAND} -E touch ${TTNN_INSTALL_DIR}/.metalium-dev-installed
  COMMENT "Installing metalium-dev component"
)

# Install ttnn-runtime component
add_custom_command(
  OUTPUT ${TTNN_INSTALL_DIR}/.ttnn-runtime-installed
  DEPENDS ${TTNN_INSTALL_DIR}/.metalium-runtime-installed
  COMMAND cmake --install ${TT_METAL_BUILD_HOME} --prefix ${TTNN_INSTALL_DIR} --component ttnn-runtime
  COMMAND ${CMAKE_COMMAND} -E touch ${TTNN_INSTALL_DIR}/.ttnn-runtime-installed
  COMMENT "Installing ttnn-runtime component"
)

# Install ttnn-dev component
add_custom_command(
  OUTPUT ${TTNN_INSTALL_DIR}/.ttnn-dev-installed
  DEPENDS ${TTNN_INSTALL_DIR}/.ttnn-runtime-installed
  COMMAND cmake --install ${TT_METAL_BUILD_HOME} --prefix ${TTNN_INSTALL_DIR} --component ttnn-dev
  COMMAND ${CMAKE_COMMAND} -E touch ${TTNN_INSTALL_DIR}/.ttnn-dev-installed
  COMMENT "Installing ttnn-dev component"
)

# Debug: Install all available components to see what's available
add_custom_command(
  OUTPUT ${TTNN_INSTALL_DIR}/.all-components-installed
  DEPENDS ${TTNN_INSTALL_DIR}
  COMMAND cmake --install ${TT_METAL_BUILD_HOME} --prefix ${TTNN_INSTALL_DIR}
  COMMAND ${CMAKE_COMMAND} -E touch ${TTNN_INSTALL_DIR}/.all-components-installed
  COMMENT "DEBUG: Installing all available components"
)

# Set of workarounds to copy around missing cpp/hpp files until ttnn/metalium install scripts are fixed
add_custom_command(
  OUTPUT ${TTNN_INSTALL_DIR}/.missing-headers-installed
  DEPENDS ${TTNN_INSTALL_DIR}/.ttnn-dev-installed
  # Copy tracy
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${TT_METAL_BUILD_HOME}/../tt_metal/third_party/tracy/public/tracy ${TTNN_INSTALL_DIR}/include/tracy
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${TT_METAL_BUILD_HOME}/../tt_metal/third_party/tracy/public/common ${TTNN_INSTALL_DIR}/include/common
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${TT_METAL_BUILD_HOME}/../tt_metal/third_party/tracy/public/client ${TTNN_INSTALL_DIR}/include/client
  # Copy op-related missing hpp files
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${TT_METAL_BUILD_HOME}/../ttnn/cpp/ttnn/operations -DDEST_DIR=${TTNN_INSTALL_DIR}/include/ttnn/operations -P ${CMAKE_CURRENT_SOURCE_DIR}/CopyHppFiles.cmake
  # Copy kernel-related missing hpp/cpp files (needed for kernel compilation in runtime)
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${TT_METAL_BUILD_HOME}/../ttnn/cpp/ttnn/operations -DDEST_DIR=${TTNN_INSTALL_DIR}/libexec/tt-metalium/ttnn/cpp/ttnn/operations -P ${CMAKE_CURRENT_SOURCE_DIR}/CopyHppFiles.cmake
  COMMAND ${CMAKE_COMMAND} -DSOURCE_DIR=${TT_METAL_BUILD_HOME}/../ttnn/cpp/ttnn/operations -DDEST_DIR=${TTNN_INSTALL_DIR}/libexec/tt-metalium/ttnn/cpp/ttnn/operations -P ${CMAKE_CURRENT_SOURCE_DIR}/CopyCppFiles.cmake
  COMMAND ${CMAKE_COMMAND} -E copy_directory ${TT_METAL_BUILD_HOME}/../tt_metal/api/tt-metalium ${TTNN_INSTALL_DIR}/libexec/tt-metalium/tt_metal/api/tt-metalium
  #
  COMMAND ${CMAKE_COMMAND} -E touch ${TTNN_INSTALL_DIR}/.missing-headers-installed
  COMMENT "Installing missing headers for precompiled header"
)

# Target to install all components
add_custom_target(install-ttnn-components
  DEPENDS
    ${TTNN_INSTALL_DIR}/.metalium-runtime-installed
    ${TTNN_INSTALL_DIR}/.metalium-dev-installed
    ${TTNN_INSTALL_DIR}/.ttnn-runtime-installed
    ${TTNN_INSTALL_DIR}/.ttnn-dev-installed
    ${TTNN_INSTALL_DIR}/.missing-headers-installed
    ${TTNN_INSTALL_DIR}/.all-components-installed
  COMMENT "Installing all required TT-NN components"
)
