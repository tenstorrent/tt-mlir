include(${CMAKE_SOURCE_DIR}/cmake/SystemIncludes.cmake)

include(FetchContent)
FetchContent_Declare(
    tt-logger
    GIT_REPOSITORY https://github.com/tenstorrent/tt-logger.git
    GIT_TAG        v1.1.8
)
set(SPDLOG_BUILD_PIC ON CACHE INTERNAL "")
FetchContent_MakeAvailable(tt-logger)

mark_as_system_includes(tt-logger spdlog_header_only fmt-header-only)
