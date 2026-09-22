# ONNX Runtime C API headers download.
# Invokes git commands manually, since cmake does not support sparse checkout.
# Initializes empty repo (with branch and provided tag) and checks out headers from HEAD.
set(TT_CRANK_ORT_REPOSITORY "https://github.com/microsoft/onnxruntime.git")
set(TT_CRANK_ORT_VERSION "1.30.0")
set(TT_CRANK_ORT_HEADERS_PATH "include/onnxruntime/core/session")

find_package(Git REQUIRED)
include(ExternalProject)
ExternalProject_Add(onnxruntime_c_api_headers
    PREFIX "${CMAKE_CURRENT_BINARY_DIR}"
    DOWNLOAD_COMMAND ${CMAKE_COMMAND} -E rm -rf <SOURCE_DIR>
        COMMAND ${GIT_EXECUTABLE} clone --quiet --no-checkout --filter=blob:none --depth 1
            --branch v${TT_CRANK_ORT_VERSION} ${TT_CRANK_ORT_REPOSITORY} <SOURCE_DIR>
        COMMAND ${GIT_EXECUTABLE} -C <SOURCE_DIR> checkout --quiet HEAD --
            ${TT_CRANK_ORT_HEADERS_PATH}/onnxruntime_c_api.h
            ${TT_CRANK_ORT_HEADERS_PATH}/onnxruntime_ep_c_api.h
            ${TT_CRANK_ORT_HEADERS_PATH}/onnxruntime_error_code.h
    CONFIGURE_COMMAND ""
    BUILD_COMMAND ""
    INSTALL_COMMAND ""
)
ExternalProject_Get_Property(onnxruntime_c_api_headers SOURCE_DIR)

# Consumers should link this.
add_library(onnxruntime_c_api INTERFACE)
target_include_directories(onnxruntime_c_api SYSTEM INTERFACE "${SOURCE_DIR}/${TT_CRANK_ORT_HEADERS_PATH}")
add_dependencies(onnxruntime_c_api onnxruntime_c_api_headers)
