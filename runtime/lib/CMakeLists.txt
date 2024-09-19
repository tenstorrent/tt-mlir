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
  add_library(TTRuntimeTTMetal INTERFACE)
endif()

message(STATUS "Runtimes Enabled: TTNN[${TT_RUNTIME_ENABLE_TTNN}] TTMETAL[${TT_RUNTIME_ENABLE_TTMETAL}]")

add_library(TTBinary STATIC binary.cpp)
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
endif()

add_library(TTRuntime STATIC runtime.cpp)
set_property(TARGET TTRuntime PROPERTY CXX_STANDARD 20)
if (TTMLIR_ENABLE_RUNTIME AND TT_RUNTIME_ENABLE_TTNN)
  target_compile_definitions(TTRuntime PUBLIC TT_RUNTIME_ENABLE_TTNN)
endif()
if (TTMLIR_ENABLE_RUNTIME AND TT_RUNTIME_ENABLE_TTMETAL)
  target_compile_definitions(TTRuntime PUBLIC TT_RUNTIME_ENABLE_TTMETAL)
endif()

target_include_directories(TTRuntime
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)

target_link_libraries(TTRuntime
  PRIVATE
    TTBinary
    TTRuntimeSysDesc
    TTRuntimeTTNN
    TTRuntimeTTMetal
    TTRuntimeDebug
)

add_dependencies(TTRuntime TTBinary TTRuntimeSysDesc TTRuntimeDebug FBS_GENERATION)

if (TTMLIR_ENABLE_RUNTIME)
  set_target_properties(TTRuntime PROPERTIES PUBLIC_HEADER "../include/tt/runtime/runtime.h;../include/tt/runtime/types.h;../include/tt/runtime/utils.h")
  install(TARGETS TTRuntime
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/tt/runtime
        COMPONENT SharedLib
  )
endif()
