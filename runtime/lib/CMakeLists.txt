if (TTMLIR_ENABLE_RUNTIME)
  if (TT_RUNTIME_ENABLE_TTNN)
    add_subdirectory(ttnn)
  else()
    add_library(TTRuntimeTTNN INTERFACE)
  endif()
  if (TT_RUNTIME_ENABLE_TTMETAL)
    add_subdirectory(ttmetal)
  else()
    add_library(TTRuntimeTTMetal INTERFACE)
  endif()
else()
  add_library(TTRuntimeTTNN INTERFACE)
  add_library(TTRuntimeTTNNTypes INTERFACE)
  add_library(TTRuntimeTTNNUtils INTERFACE)
  add_library(TTRuntimeTTMetal INTERFACE)
endif()

message(STATUS "Runtimes Enabled: TTNN[${TT_RUNTIME_ENABLE_TTNN}] TTMETAL[${TT_RUNTIME_ENABLE_TTMETAL}]")

add_library(TTBinary STATIC binary.cpp)
set_property(TARGET TTBinary PROPERTY CXX_STANDARD 20)
target_include_directories(TTBinary
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
add_dependencies(TTBinary FBS_GENERATION)

if (TTMLIR_ENABLE_RUNTIME AND (TT_RUNTIME_ENABLE_TTNN OR TT_RUNTIME_ENABLE_TTMETAL))
  add_subdirectory(common)
else()
  add_library(TTRuntimeSysDesc INTERFACE)
  add_library(TTRuntimeDebug INTERFACE)
  add_library(TTRuntimeWorkarounds INTERFACE)
endif()

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

target_link_libraries(TTMLIRRuntime
  PRIVATE
    TTBinary
    TTRuntimeSysDesc
    TTRuntimeTTNN
    TTRuntimeTTMetal
    TTRuntimeDebug
    TTRuntimeWorkarounds
    ${TTNN_LIBS}
    flatbuffers
)

set_target_properties(TTMLIRRuntime PROPERTIES INSTALL_RPATH "$ORIGIN")
set_target_properties(TTMLIRRuntime PROPERTIES BUILD_WITH_INSTALL_RPATH TRUE)

add_dependencies(TTMLIRRuntime TTBinary TTRuntimeSysDesc TTRuntimeDebug TTRuntimeWorkarounds FBS_GENERATION)

if (TTMLIR_ENABLE_RUNTIME)
  set_target_properties(TTMLIRRuntime PROPERTIES PUBLIC_HEADER "../include/tt/runtime/runtime.h;../include/tt/runtime/types.h;../include/tt/runtime/utils.h")
  install(TARGETS TTMLIRRuntime
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt/runtime
        COMPONENT SharedLib
  )
  install(TARGETS TTMLIRRuntime
        LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}
        COMPONENT SharedLib
  )
endif()
