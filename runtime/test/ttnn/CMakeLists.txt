if (NOT TTMLIR_ENABLE_RUNTIME_TESTS OR NOT TT_RUNTIME_ENABLE_TTNN)
  add_library(TTRuntimeTTNNTestLib INTERFACE)
  return()
endif()

add_library(TTRuntimeTTNNTestLib
  STATIC
  utils.cpp
  dylib.cpp
)
set_property(TARGET TTRuntimeTTNNTestLib PROPERTY CXX_STANDARD 20)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|x86|i386|i686)$")
  target_compile_options(TTRuntimeTTNNTestLib PUBLIC -march=x86-64-v3)
endif()
target_include_directories(TTRuntimeTTNNTestLib PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/runtime/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNTestLib SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeTTNNTestLib TTRuntimeTTNNUtils FBS_GENERATION)
target_link_libraries(TTRuntimeTTNNTestLib PUBLIC TTRuntimeTTNNUtils)

set(TTMLIR_RUNTIME_TEST_TTNN_PUBLIC_HEADERS
  "${PROJECT_SOURCE_DIR}/runtime/include/tt/runtime/test/ttnn/dylib.h"
)
set_target_properties(TTRuntimeTTNNTestLib PROPERTIES PUBLIC_HEADER "${TTMLIR_RUNTIME_TEST_TTNN_PUBLIC_HEADERS}")
include(GNUInstallDirs)
install(TARGETS TTRuntimeTTNNTestLib
  COMPONENT TTNNStandalone
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt/runtime/test
)

add_subdirectory(gtest)
