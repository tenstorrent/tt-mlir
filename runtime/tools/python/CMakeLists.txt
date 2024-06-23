file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/ DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

add_custom_target(ttrt
  COMMAND TTMLIR_ENABLE_RUNTIME=${TTMLIR_ENABLE_RUNTIME} SOURCE_ROOT=${TTMLIR_SOURCE_DIR} python -m pip install .
  WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
  COMMENT "python ttrt package"
)
if (TTMLIR_ENABLE_RUNTIME)
  if (NOT TTMLIR_DISABLE_RUNTIME_TTNN)
    add_dependencies(ttrt TTRuntimeTTNN)
  endif()
  if (NOT TTMLIR_DISABLE_RUNTIME_TTMETAL)
    add_dependencies(ttrt TTRuntimeTTMetal)
  endif()
endif()
