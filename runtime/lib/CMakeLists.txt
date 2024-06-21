add_library(TTRuntimeBinary STATIC binary/binary.cpp)
target_include_directories(TTRuntimeBinary PUBLIC ${PROJECT_SOURCE_DIR}/runtime/include ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common)
add_dependencies(TTRuntimeBinary FBS_GENERATION)

if(TTMLIR_ENABLE_RUNTIME)
  if (NOT TTMLIR_DISABLE_RUNTIME_TTNN)
    add_library(TTRuntimeTTNN STATIC ttnn/runtime.cpp)
    target_include_directories(TTRuntimeTTNN PUBLIC
      ${PROJECT_SOURCE_DIR}/runtime/include
      ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common
    )
    target_include_directories(TTRuntimeTTNN PUBLIC "$<BUILD_INTERFACE:${TTMETAL_INCLUDE_DIRS}>")
    target_link_directories(TTRuntimeTTNN PUBLIC ${TTMETAL_LIBRARY_DIR})
    target_link_libraries(TTRuntimeTTNN
      PUBLIC
      ${TTNN_LIBRARY}
    )
    add_dependencies(TTRuntimeTTNN TTRuntimeBinary tt-metal)
  endif()
  if (NOT TTMLIR_DISABLE_RUNTIME_TTMETAL)
    add_library(TTRuntimeTTMetal STATIC ttmetal/runtime.cpp)
    target_include_directories(TTRuntimeTTMetal PUBLIC ${PROJECT_SOURCE_DIR}/runtime/include ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common)
    add_dependencies(TTRuntimeTTMetal TTRuntimeBinary tt-metal)
  endif()
endif()
