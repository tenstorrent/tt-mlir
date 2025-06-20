# InstallComponents.cmake
# Handles installation of required TT-NN components for cpp standalone builds

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
  DEPENDS ${TTNN_INSTALL_DIR}/.ttnn-dev-installed
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
  DEPENDS ${TTNN_INSTALL_DIR}
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

# # Debug: Install all available components to see what's available
# add_custom_command(
#   OUTPUT ${TTNN_INSTALL_DIR}/.all-components-installed
#   DEPENDS ${TTNN_INSTALL_DIR}/.metalium-dev-installed
#   COMMAND cmake --install ${TT_METAL_BUILD_HOME} --prefix ${TTNN_INSTALL_DIR}
#   COMMAND ${CMAKE_COMMAND} -E touch ${TTNN_INSTALL_DIR}/.all-components-installed
#   COMMENT "DEBUG: Installing all available components"
# )

# Target to install all components
add_custom_target(install-ttnn-components
  DEPENDS
    ${TTNN_INSTALL_DIR}/.metalium-runtime-installed
    ${TTNN_INSTALL_DIR}/.metalium-dev-installed
    ${TTNN_INSTALL_DIR}/.ttnn-runtime-installed
    ${TTNN_INSTALL_DIR}/.ttnn-dev-installed
    # ${TTNN_INSTALL_DIR}/.all-components-installed
  COMMENT "Installing all required TT-NN components"
)
