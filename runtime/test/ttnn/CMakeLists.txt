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
target_compile_options(TTRuntimeTTNNTestLib PUBLIC -mavx -mavx2)
target_include_directories(TTRuntimeTTNNTestLib PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/ttnn/include
  ${PROJECT_SOURCE_DIR}/runtime/lib/test/include
  ${PROJECT_SOURCE_DIR}/runtime/test/include
  ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_include_directories(TTRuntimeTTNNTestLib SYSTEM PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
add_dependencies(TTRuntimeTTNNTestLib TTRuntimeTTNNUtils FBS_GENERATION)
target_link_libraries(TTRuntimeTTNNTestLib PUBLIC TTRuntimeTTNNUtils)

set(TTMLIR_RUNTIME_TEST_PUBLIC_HEADERS
  "../include/tt/runtime/ttnn/test/dylib.h"
)
set_target_properties(TTRuntimeTTNNTestLib PROPERTIES PUBLIC_HEADER "${TTMLIR_RUNTIME_TEST_PUBLIC_HEADERS}")
install(TARGETS TTRuntimeTTNNTestLib
  PUBLIC_HEADER
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt/runtime/test
    COMPONENT TTNNStandalone
)

add_subdirectory(gtest)
