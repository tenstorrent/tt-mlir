include(${PROJECT_SOURCE_DIR}/cmake/SystemIncludes.cmake)

set(CPM_SOURCE_CACHE "${PROJECT_BINARY_DIR}/.cpmcache" CACHE PATH "CPM source cache for tt-crank third-party sources")

include(FetchContent)
FetchContent_Declare(
    tt-logger
    GIT_REPOSITORY https://github.com/tenstorrent/tt-logger.git
    GIT_TAG        v1.1.8
)
set(SPDLOG_BUILD_PIC ON CACHE INTERNAL "")
FetchContent_MakeAvailable(tt-logger)

mark_as_system_includes(tt-logger spdlog spdlog_header_only fmt fmt-header-only)
