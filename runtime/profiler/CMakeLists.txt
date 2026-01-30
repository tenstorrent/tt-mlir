add_library(TTMLIRProfiler
  STATIC
  profiler_impl.cpp
)

target_include_directories(TTMLIRProfiler PUBLIC
  ${PROJECT_SOURCE_DIR}/runtime/profiler
)

set(TRACY_CSV_EXPORT "${TRACY_BINARY_DIR}/csvexport-release")
set(TRACY_CAPTURE "${TRACY_BINARY_DIR}/capture-release")

add_custom_target(copy_tracy_tools
    COMMAND ${CMAKE_COMMAND} -E copy
        "${TRACY_CSV_EXPORT}"
        "${TRACY_CAPTURE}"
        "${CMAKE_CURRENT_BINARY_DIR}"
    BYPRODUCTS
        "${CMAKE_CURRENT_BINARY_DIR}/csvexport-release"
        "${CMAKE_CURRENT_BINARY_DIR}/capture-release"
)

add_dependencies(copy_tracy_tools TRACY_LIBRARY)
add_dependencies(TTMLIRProfiler TRACY_LIBRARY copy_tracy_tools)
