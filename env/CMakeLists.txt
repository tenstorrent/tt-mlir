cmake_minimum_required(VERSION 3.20.0)
project(ttmlir-toolchain LANGUAGES CXX C)
find_package(Python3 3.12 REQUIRED COMPONENTS Interpreter Development.Module)

set(FLATBUFFERS_VERSION "fb9afbafc7dfe226b9db54d4923bfb8839635274")
set(LLVM_PROJECT_VERSION "22.1.0")
set(STABLEHLO_VERSION "0a4440a5c8de45c4f9649bf3eb4913bf3f97da0d")
set(SHARDY_VERSION "edfd6730ddfc39da5fbea8b6b202357fdf1cdb90")
set(LLVM_BUILD_TYPE MinSizeRel CACHE STRING "Build type for LLVM")

include(ExternalProject)
string(REPLACE ";" "," TTMLIR_EXTERNAL_CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH}")

set(TTMLIR_TOOLCHAIN_DIR "$ENV{TTMLIR_TOOLCHAIN_DIR}" CACHE PATH
    "Install prefix for the tt-mlir toolchain")
if(NOT TTMLIR_TOOLCHAIN_DIR)
    message(FATAL_ERROR "TTMLIR_TOOLCHAIN_DIR must name an existing install prefix")
endif()
message(STATUS "TTMLIR_TOOLCHAIN_DIR: ${TTMLIR_TOOLCHAIN_DIR}")

if(NOT EXISTS "${TTMLIR_TOOLCHAIN_DIR}")
    message(FATAL_ERROR "The directory ${TTMLIR_TOOLCHAIN_DIR} does not exist")
endif()


ExternalProject_Add(
    flatbuffers
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${TTMLIR_TOOLCHAIN_DIR}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DFLATBUFFERS_BUILD_TESTS=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON # Required for linking into shared libraries (python bindings)
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG ${FLATBUFFERS_VERSION}
    GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
    UPDATE_DISCONNECTED ON
)

set(TTMLIR_BUILD_LLVM ON CACHE BOOL "Enable LLVM Build")
if(TTMLIR_BUILD_LLVM)
    # -DMLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS=ON set if debug
    ExternalProject_Add(
        llvm-project
        PREFIX ${TTMLIR_TOOLCHAIN_DIR}

        URL
            "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LLVM_PROJECT_VERSION}/llvm-project-${LLVM_PROJECT_VERSION}.src.tar.xz"
        # Strongly recommended:
        URL_HASH SHA256=25d2e2adc4356d758405dd885fcfd6447bce82a90eb78b6b87ce0934bd077173
        DOWNLOAD_EXTRACT_TIMESTAMP TRUE
        PATCH_COMMAND git apply "${CMAKE_CURRENT_LIST_DIR}/patches/affine-allow-symbol-vars.patch"
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${LLVM_BUILD_TYPE}
        -DPython3_EXECUTABLE=${Python3_EXECUTABLE}
        -DCMAKE_INSTALL_PREFIX=${TTMLIR_TOOLCHAIN_DIR}
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_C_COMPILER_LAUNCHER=${CMAKE_C_COMPILER_LAUNCHER}
        -DCMAKE_CXX_COMPILER_LAUNCHER=${CMAKE_CXX_COMPILER_LAUNCHER}
        -DCMAKE_PREFIX_PATH=${TTMLIR_EXTERNAL_CMAKE_PREFIX_PATH}
        -DLLVM_ENABLE_PROJECTS=mlir,lld
        -DLLVM_TARGETS_TO_BUILD=host
        -DLLVM_INSTALL_UTILS=ON
        -DLLVM_INSTALL_GTEST=ON
        -DLLVM_ENABLE_ASSERTIONS=ON
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON
        -DCMAKE_C_FLAGS=-D_LIBCPP_HAS_NO_LIBRARY_ALIGNED_ALLOCATION
        SOURCE_SUBDIR llvm
        LIST_SEPARATOR ","
        UPDATE_COMMAND ""
        UPDATE_DISCONNECTED ON
    )

endif()

ExternalProject_Add(stablehlo
   PREFIX ${TTMLIR_TOOLCHAIN_DIR}
   GIT_REPOSITORY https://github.com/openxla/stablehlo.git
   GIT_TAG ${STABLEHLO_VERSION}
   GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
   UPDATE_DISCONNECTED ON
   CONFIGURE_COMMAND ""
   BUILD_COMMAND ""
   INSTALL_COMMAND ""
   PATCH_COMMAND git config user.email "tt-mlir@tenstorrent.com" && git config user.name "tenstorrent" && git apply --index "${CMAKE_CURRENT_LIST_DIR}/patches/stablehlo-complex-mul-expander.patch" && git commit -m "tt-mlir related patch" && git apply --index "${CMAKE_CURRENT_LIST_DIR}/patches/stablehlo-token-type-qualify.patch" && git commit -m "tt-mlir related patch"
)

ExternalProject_Add(shardy
   PREFIX ${TTMLIR_TOOLCHAIN_DIR}
   GIT_REPOSITORY https://github.com/openxla/shardy.git
   GIT_TAG ${SHARDY_VERSION}
   GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
   UPDATE_DISCONNECTED ON
   CONFIGURE_COMMAND ""
   BUILD_COMMAND ""
   INSTALL_COMMAND ""
   PATCH_COMMAND git config user.email "tt-mlir@tenstorrent.com" && git config user.name "tenstorrent" && git apply --index "${CMAKE_CURRENT_LIST_DIR}/patches/shardy.patch" && git commit -m "tt-mlir related patch" && git apply --index "${CMAKE_CURRENT_LIST_DIR}/patches/shardy_mpmd_pybinds.patch" && git commit -m "tt-mlir related patch"
)

if(TTMLIR_BUILD_LLVM)
    add_custom_target(llvm-lit ALL COMMAND cp ${TTMLIR_TOOLCHAIN_DIR}/src/llvm-project-build/bin/llvm-lit ${TTMLIR_TOOLCHAIN_DIR}/bin/llvm-lit DEPENDS llvm-project)
    add_custom_target(run-clang-tidy-install ALL COMMAND cp ${TTMLIR_TOOLCHAIN_DIR}/src/llvm-project/clang-tools-extra/clang-tidy/tool/run-clang-tidy.py ${TTMLIR_TOOLCHAIN_DIR}/bin/run-clang-tidy.py DEPENDS llvm-project)
endif()
