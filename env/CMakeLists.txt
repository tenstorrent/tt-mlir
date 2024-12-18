cmake_minimum_required(VERSION 3.20.0)
project(ttmlir-toolchain LANGUAGES CXX C)

set(FLATBUFFERS_VERSION "fb9afbafc7dfe226b9db54d4923bfb8839635274")
set(LLVM_PROJECT_VERSION "e813750354bbc08551cf23ff559a54b4a9ea1f29")
set(STABLEHLO_VERSION "d40285ef3db0687e3f1e2bb0d716d748485a9739")
set(SHARDY_VERSION "55f44c23b766be38bccb0b2394b0e8dfba45694e")

include(ExternalProject)

if (NOT DEFINED ENV{TTMLIR_TOOLCHAIN_DIR})
    message(FATAL_ERROR "Please set TTMLIR_TOOLCHAIN_DIR to the directory where the toolchain will be installed")
else()
    set(TTMLIR_TOOLCHAIN_DIR $ENV{TTMLIR_TOOLCHAIN_DIR})
endif()

if (NOT EXISTS "${TTMLIR_TOOLCHAIN_DIR}")
    message( FATAL_ERROR "The directory ${TTMLIR_TOOLCHAIN_DIR} does not exist. Please create it before running this script.\n  sudo mkdir -p ${TTMLIR_TOOLCHAIN_DIR}\n  sudo chown -R $ENV{USER} ${TTMLIR_TOOLCHAIN_DIR}")
endif()


add_custom_target(python-venv ALL COMMAND TTMLIR_TOOLCHAIN=${TTMLIR_TOOLCHAIN_DIR} CURRENT_SOURCE_DIR=${CMAKE_CURRENT_SOURCE_DIR} bash ${CMAKE_CURRENT_SOURCE_DIR}/build_venv.sh)

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
    DEPENDS python-venv
)

# -DMLIR_ENABLE_EXPENSIVE_PATTERN_API_CHECKS=ON set if debug
ExternalProject_Add(
    llvm-project
    # Super hacky way to install the python dependencies before the build
    PATCH_COMMAND bash -c "source ${CMAKE_CURRENT_SOURCE_DIR}/activate && pip install -r mlir/python/requirements.txt"
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
    -DPython3_FIND_VIRTUALENV=ONLY
    -DPython3_EXECUTABLE=${TTMLIR_TOOLCHAIN_DIR}/venv/bin/python
    -DCMAKE_INSTALL_PREFIX=${TTMLIR_TOOLCHAIN_DIR}
    -DCMAKE_C_COMPILER=clang
    -DCMAKE_CXX_COMPILER=clang++
    -DLLVM_ENABLE_PROJECTS=mlir
    -DLLVM_INSTALL_UTILS=ON
    # Build shared libraries
    # ======================
    -DLLVM_ENABLE_PIC=ON
    -DLLVM_BUILD_LLVM_DYLIB=ON
    -DLLVM_INSTALL_GTEST=ON
    -DLLVM_LINK_LLVM_DYLIB=ON
    -DMLIR_BUILD_MLIR_C_DYLIB=ON
    -DMLIR_LINK_MLIR_DYLIB=ON
    -DMLIR_BUILD_MLIR_C_DYLIB=ON
    # ======================
    -DCMAKE_BUILD_TYPE=Debug
    -DLLVM_ENABLE_ASSERTIONS=ON
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    -DCMAKE_C_FLAGS=-D_LIBCPP_HAS_NO_LIBRARY_ALIGNED_ALLOCATION
    SOURCE_SUBDIR llvm
    LIST_SEPARATOR ","
    GIT_REPOSITORY https://github.com/llvm/llvm-project.git
    GIT_TAG ${LLVM_PROJECT_VERSION}
    GIT_PROGRESS ON
    DEPENDS python-venv
)

ExternalProject_Add(stablehlo
   PREFIX ${TTMLIR_TOOLCHAIN_DIR}
   GIT_REPOSITORY https://github.com/openxla/stablehlo.git
   GIT_TAG ${STABLEHLO_VERSION}
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
   PATCH_COMMAND git config user.email "tt-mlir@tenstorrent.com" && git config user.name "tenstorrent" && git apply --index "${CMAKE_CURRENT_LIST_DIR}/patches/shardy.patch" && git commit -m "tt-mlir related patch"
)

add_custom_target(llvm-lit ALL COMMAND cp llvm-project-prefix/src/llvm-project-build/bin/llvm-lit ${TTMLIR_TOOLCHAIN_DIR}/bin/llvm-lit DEPENDS llvm-project)
add_custom_target(run-clang-tidy-install ALL COMMAND cp llvm-project-prefix/src/llvm-project/clang-tools-extra/clang-tidy/tool/run-clang-tidy.py ${TTMLIR_TOOLCHAIN_DIR}/bin/run-clang-tidy.py DEPENDS llvm-project)
