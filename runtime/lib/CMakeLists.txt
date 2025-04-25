message(STATUS "Runtimes Enabled: TTNN[${TT_RUNTIME_ENABLE_TTNN}] TTMETAL[${TT_RUNTIME_ENABLE_TTMETAL}]")

add_subdirectory(common)
add_subdirectory(ttnn)
add_subdirectory(ttmetal)

add_library(TTBinary STATIC binary.cpp)
set_property(TARGET TTBinary PROPERTY CXX_STANDARD 20)
target_include_directories(TTBinary
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
add_dependencies(TTBinary FBS_GENERATION)

add_library(TTMLIRRuntime SHARED runtime.cpp)
set_property(TARGET TTMLIRRuntime PROPERTY CXX_STANDARD 20)
if (TTMLIR_ENABLE_RUNTIME AND TT_RUNTIME_ENABLE_TTNN)
  target_compile_definitions(TTMLIRRuntime PUBLIC TT_RUNTIME_ENABLE_TTNN)
endif()
if (TTMLIR_ENABLE_RUNTIME AND TT_RUNTIME_ENABLE_TTMETAL)
  target_compile_definitions(TTMLIRRuntime PUBLIC TT_RUNTIME_ENABLE_TTMETAL)
endif()

target_include_directories(TTMLIRRuntime
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)

set(TTNN_LIBS TTMETAL_LIBRARY DEVICE_LIBRARY TTNN_LIBRARY)
if (TT_RUNTIME_ENABLE_PERF_TRACE)
  list(APPEND TTNN_LIBS TRACY_LIBRARY)
endif()

target_link_directories(TTMLIRRuntime PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib)
target_link_directories(TTMLIRRuntime PRIVATE ${TTMLIR_TOOLCHAIN_DIR}/lib64)

target_link_libraries(TTMLIRRuntime
  PRIVATE
    TTBinary
    TTRuntimeSysDesc
    TTRuntimeTTNN
    TTRuntimeTTMetal
    TTRuntimeDebug
    TTRuntimeDylibs
    ${TTNN_LIBS}
    flatbuffers
)

target_link_libraries(TTMLIRRuntime PUBLIC coverage_config)

set_target_properties(TTMLIRRuntime PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(TTMLIRRuntime PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

add_dependencies(TTMLIRRuntime TTBinary TTRuntimeSysDesc TTRuntimeDebug TTRuntimeDylibs TTRuntimeTTNN TTRuntimeTTMetal FBS_GENERATION)

if (TTMLIR_ENABLE_RUNTIME)
  set(TTMLIR_RUNTIME_PUBLIC_HEADERS
    "../include/tt/runtime/runtime.h"
    "../include/tt/runtime/types.h"
    "../include/tt/runtime/utils.h"
    "../include/tt/runtime/workarounds.h"
    "../include/tt/runtime/tensor_cache.h"
  )
  set_target_properties(TTMLIRRuntime PROPERTIES PUBLIC_HEADER "${TTMLIR_RUNTIME_PUBLIC_HEADERS}")
  install(TARGETS TTMLIRRuntime
    PUBLIC_HEADER
      DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt/runtime
      COMPONENT SharedLib
    LIBRARY
      DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
      COMPONENT SharedLib
  )
endif()
