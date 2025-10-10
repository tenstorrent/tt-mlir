cmake_minimum_required(VERSION 3.20.0)
project(ttmlir-toolchain LANGUAGES CXX C)

set(FLATBUFFERS_VERSION "fb9afbafc7dfe226b9db54d4923bfb8839635274")
set(LLVM_PROJECT_VERSION "f8cb7987c64dcffb72414a40560055cb717dbf74")
set(STABLEHLO_VERSION "e07debd5e257ec1e118f18c54068977b89f03b2f")
set(SHARDY_VERSION "a8f93f165dd5e56a44c300ed68bb5233c43dd30c")
set(LLVM_BUILD_TYPE MinSizeRel CACHE STRING "Build type for LLVM")

include(ExternalProject)

set(ENV{_ACTIVATE_ECHO_TOOLCHAIN_DIR_AND_EXIT} "1")
execute_process(
    COMMAND bash activate
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE TTMLIR_TOOLCHAIN_DIR
    RESULT_VARIABLE ACTIVATE_RESULT
)

if (ACTIVATE_RESULT)
    message(FATAL_ERROR "Failed to get toolchain environment")
endif()

set(ENV{TTMLIR_TOOLCHAIN_DIR} "${TTMLIR_TOOLCHAIN_DIR}")
message(STATUS "TTMLIR_TOOLCHAIN_DIR: ${TTMLIR_TOOLCHAIN_DIR}")

if (NOT EXISTS "${TTMLIR_TOOLCHAIN_DIR}")
    message( FATAL_ERROR "The directory ${TTMLIR_TOOLCHAIN_DIR} does not exist. Please create it before running this script.\n  sudo mkdir -p ${TTMLIR_TOOLCHAIN_DIR}\n  sudo chown -R $ENV{USER} ${TTMLIR_TOOLCHAIN_DIR}")
endif()

add_custom_target(python-venv ALL COMMAND TTMLIR_TOOLCHAIN=${TTMLIR_TOOLCHAIN_DIR} ENV_DIR=${CMAKE_CURRENT_SOURCE_DIR} bash ${CMAKE_CURRENT_SOURCE_DIR}/init_venv.sh)

ExternalProject_Add(
    flatbuffers
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX=${TTMLIR_TOOLCHAIN_DIR}
    -DFLATBUFFERS_BUILD_TESTS=OFF
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON # Required for linking into shared libraries (python bindings)
    GIT_REPOSITORY https://github.com/google/flatbuffers.git
    GIT_TAG ${FLATBUFFERS_VERSION}
    GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
    UPDATE_DISCONNECTED ON
    DEPENDS python-venv
)

set(TTMLIR_BUILD_LLVM ON CACHE BOOL "Enable LLVM Build")
if(TTMLIR_BUILD_LLVM)
    # -DMLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS=ON set if debug
    ExternalProject_Add(
        llvm-project
        # Super hacky way to install the python dependencies before the build
        PATCH_COMMAND git config user.email "tt-mlir@tenstorrent.com" && git config user.name "tenstorrent" && git apply --index "${CMAKE_CURRENT_LIST_DIR}/patches/llvm.patch" && git commit -m "tt-mlir related patch" && bash -c "source ${CMAKE_CURRENT_SOURCE_DIR}/activate && pip install -r mlir/python/requirements.txt"
        CMAKE_GENERATOR Ninja
        CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${LLVM_BUILD_TYPE}
        -DPython3_FIND_VIRTUALENV=ONLY
        -DPython3_EXECUTABLE=${TTMLIR_TOOLCHAIN_DIR}/venv/bin/python
        -DCMAKE_INSTALL_PREFIX=${TTMLIR_TOOLCHAIN_DIR}
        -DCMAKE_C_COMPILER=clang
        -DCMAKE_CXX_COMPILER=clang++
        -DLLVM_ENABLE_PROJECTS=mlir
        -DLLVM_INSTALL_UTILS=ON
        -DLLVM_INSTALL_GTEST=ON
        -DLLVM_ENABLE_ASSERTIONS=ON
        -DMLIR_ENABLE_BINDINGS_PYTHON=ON
        -DCMAKE_C_FLAGS=-D_LIBCPP_HAS_NO_LIBRARY_ALIGNED_ALLOCATION
        SOURCE_SUBDIR llvm
        LIST_SEPARATOR ","
        GIT_REPOSITORY https://github.com/llvm/llvm-project.git
        GIT_TAG ${LLVM_PROJECT_VERSION}
        GIT_PROGRESS OFF
        GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
        UPDATE_DISCONNECTED ON
        DEPENDS python-venv
    )
endif()

ExternalProject_Add(stablehlo
   PREFIX ${TTMLIR_TOOLCHAIN_DIR}
   GIT_REPOSITORY https://github.com/openxla/stablehlo.git
   GIT_TAG ${STABLEHLO_VERSION}
   GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
   CONFIGURE_COMMAND ""
   BUILD_COMMAND ""
   INSTALL_COMMAND ""
)

ExternalProject_Add(shardy
   PREFIX ${TTMLIR_TOOLCHAIN_DIR}
   GIT_REPOSITORY https://github.com/openxla/shardy.git
   GIT_TAG ${SHARDY_VERSION}
   GIT_REMOTE_UPDATE_STRATEGY CHECKOUT
   CONFIGURE_COMMAND ""
   BUILD_COMMAND ""
   INSTALL_COMMAND ""
   PATCH_COMMAND git config user.email "tt-mlir@tenstorrent.com" && git config user.name "tenstorrent" && git apply --index "${CMAKE_CURRENT_LIST_DIR}/patches/shardy.patch" && git commit -m "tt-mlir related patch" && git apply --index "${CMAKE_CURRENT_LIST_DIR}/patches/shardy_mpmd_pybinds.patch" && git commit -m "tt-mlir related patch"
)

if(TTMLIR_BUILD_LLVM)
    add_custom_target(llvm-lit ALL COMMAND cp llvm-project-prefix/src/llvm-project-build/bin/llvm-lit ${TTMLIR_TOOLCHAIN_DIR}/bin/llvm-lit DEPENDS llvm-project)
    add_custom_target(run-clang-tidy-install ALL COMMAND cp llvm-project-prefix/src/llvm-project/clang-tools-extra/clang-tidy/tool/run-clang-tidy.py ${TTMLIR_TOOLCHAIN_DIR}/bin/run-clang-tidy.py DEPENDS llvm-project)
endif()
