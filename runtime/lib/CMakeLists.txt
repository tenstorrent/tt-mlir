set(TT_RUNTIME_ENABLE_TTNN OFF)
set(TT_RUNTIME_ENABLE_TTMETAL OFF)

if (TTMLIR_ENABLE_RUNTIME)
  set(TT_RUNTIME_ENABLE_TTNN NOT TTMLIR_DISABLE_RUNTIME_TTNN)
  set(TT_RUNTIME_ENABLE_TTMETAL OFF)

  if (TT_RUNTIME_ENABLE_TTNN)
    add_library(TTRuntimeTTNN
      STATIC
      ttnn/runtime.cpp
      ttnn/program.cpp
    )
    target_include_directories(TTRuntimeTTNN PUBLIC
      ${PROJECT_SOURCE_DIR}/runtime/include
      ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
    )
    target_include_directories(TTRuntimeTTNN PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
    target_link_directories(TTRuntimeTTNN PUBLIC ${TTMETAL_LIBRARY_DIR})
    target_link_libraries(TTRuntimeTTNN PUBLIC ${TTNN_LIBRARY})
    add_dependencies(TTRuntimeTTNN tt-metal FBS_GENERATION)
  else()
    add_library(TTRuntimeTTNN INTERFACE)
  endif()
  if (TT_RUNTIME_ENABLE_TTMETAL)
    add_library(TTRuntimeTTMetal STATIC ttmetal/runtime.cpp)
    target_include_directories(TTRuntimeTTMetal PUBLIC ${PROJECT_SOURCE_DIR}/runtime/include ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common)
    add_dependencies(TTRuntimeTTMetal tt-metal FBS_GENERATION)
  else()
    add_library(TTRuntimeTTMetal INTERFACE)
  endif()
else()
  add_library(TTRuntimeTTNN INTERFACE)
  add_library(TTRuntimeTTMetal INTERFACE)
endif()

add_library(TTRuntime STATIC binary.cpp runtime.cpp)
if (TT_RUNTIME_ENABLE_TTNN)
  target_compile_definitions(TTRuntime PUBLIC TT_RUNTIME_ENABLE_TTNN)
endif()
if (TT_RUNTIME_ENABLE_TTMETAL)
  target_compile_definitions(TTRuntime PUBLIC TT_RUNTIME_ENABLE_TTMETAL)
endif()
target_include_directories(TTRuntime
  PUBLIC
    ${PROJECT_SOURCE_DIR}/runtime/include
    ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
)
target_link_libraries(TTRuntime
  PRIVATE
    TTRuntimeTTNN
    TTRuntimeTTMetal
)
add_dependencies(TTRuntime FBS_GENERATION)
