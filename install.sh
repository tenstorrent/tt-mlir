#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

# Function to display help
show_help() {
    echo "Usage: $0 [options]..."
    echo "  -h, --help                       Show this help message."
    echo "  --debug                          Set the build type as Debug."
    echo "  --release                        Set the build type as Release."
    echo "  --clean                          Remove build workspaces."
    echo "  -r, --enable-runtime             Enable the ttnn/metal runtime."
    echo "  --enable-runtime-tests           Enable runtime tests. Will implicitly enable runtime."
    echo "  -p, --enable-profiler            Enable Tracy profiler."
    echo "  --enable-op-model                Enable Op Model. Will implicitly enable runtime."
    echo "  -c, --enable-ccache              Enable ccache for the build."
    echo "  --disable-python-bindings        Disable python bindings to accelerate builds."
    echo "  --c-compiler-path                Set path to C compiler."
    echo "  --cxx-compiler-path              Set path to C++ compiler."
    echo "  --skip-tests                     Skip running check-ttmlir tests."
    echo "  --skip-ttrt                      Skip building ttrt utils."
}

clean() {
    echo "INFO: Removing build artifacts!"
    rm -rf build_Release* build_Debug* build_RelWithDebInfo* build_ASan* build_TSan* build built
    rm -rf third_party/tt-metal/src/tt-metal-build third_party/tt-metal/src/tt-metal-stamp
}

build_type="Release"
enable_runtime="OFF"
enable_runtime_tests="OFF"
enable_profiler="OFF"
enable_op_model="OFF"
enable_ccache="OFF"
disable_python_bindings="OFF"
c_compiler_path="clang-17"
cxx_compiler_path="clang++-17"
skip_tests="OFF"
skip_ttrt="OFF"

declare -a cmake_args

OPTIONS=h,c,p,r
LONGOPTIONS="
help
debug
release
clean
enable-ccache
c-compiler-path:
cxx-compiler-path:
enable-runtime
enable-runtime-tests
enable-profiler
enable-op-model
disable-python-bindings
skip-tests
skip-ttrt
"

# Flatten LONGOPTIONS into a comma-separated string for getopt
LONGOPTIONS=$(echo "$LONGOPTIONS" | tr '\n' ',' | sed 's/,$//')

# Parse the options
PARSED=$(getopt --options=$OPTIONS --longoptions=$LONGOPTIONS --name "$0" -- "$@")
if [[ $? -ne 0 ]]; then
    # If getopt has errors
    echo "INFO: Failed to parse arguments!"
    show_help
    exit 1
fi

eval set -- "$PARSED"
while true; do
    case "$1" in
        -h|--help)
            show_help;exit 0;;
        --debug)
            build_type="Debug";;
        --release)
            build_type="Release";;
        -c|--enable-ccache)
            enable_ccache="ON";;
        --c-compiler-path)
            c_compiler_path="$2";shift;;
        --cxx-compiler-path)
            cxx_compiler_path="$2";shift;;
        -r|--enable-runtime)
            enable_runtime="ON";;
        --enable-runtime-tests)
            enable_runtime="ON"; enable_runtime_tests="ON";;
        -p|--enable-profiler)
            enable_runtime="ON"; enable_profiler="ON";;
        --enable-op-model)
            enable_runtime="ON"; enable_op_model="ON";;
        --disable-python-bindings)
            disable_python_bindings="ON";;
        --skip-tests)
            skip_tests="ON";;
        --skip-ttrt)
            skip_ttrt="ON";;
        --clean)
	    clean; exit 0;;
        --)
            shift;break;;
    esac
    shift
done

# Check if there are unrecognized positional arguments left
if [[ $# -gt 0 ]]; then
    echo "ERROR: Unrecognized positional argument(s): $@"
    show_help
    exit 1
fi

# Validate the build_type
VALID_BUILD_TYPES=("Release" "Debug" "RelWithDebInfo" "ASan" "TSan")
if [[ ! " ${VALID_BUILD_TYPES[@]} " =~ " ${build_type} " ]]; then
    echo "ERROR: Invalid build type '$build_type'. Allowed values are Release, Debug, RelWithDebInfo, ASan, TSan."
    show_help
    exit 1
fi

# If build-dir is not specified
# Use build_type and enable_profiler setting to choose a default path
if [ -z "$build_dir" ]; then
    build_dir="build_$build_type"
    if [ "$enable_profiler" = "ON" ]; then
        build_dir="${build_dir}_tracy"
    fi
    # Create and link the build directory
    mkdir -p $build_dir
    ln -nsf $build_dir build
fi


if [ -z "$TTMLIR_ENV_ACTIVATED" ]; then
    source env/activate
fi

# Debug output to verify parsed options
echo "INFO: Enable ccache: $enable_ccache"
echo "INFO: Build type: $build_type"
echo "INFO: Build directory: $build_dir"
echo "INFO: Enable metal runtime: $enable_runtime"
echo "INFO: Enable metal runtime tests: $enable_runtime_tests"
echo "INFO: Enable metal perf trace: $enable_profiler"
echo "INFO: Enable op model: $enable_op_model"
echo "INFO: Disable python bindings: $disable_python_bindings"

# Prepare cmake arguments
cmake_args+=("-B" "$build_dir")
cmake_args+=("-G" "Ninja")
cmake_args+=("-DCMAKE_BUILD_TYPE=$build_type")

if [ "$c_compiler_path" != "" ]; then
    echo "INFO: C compiler: $c_compiler_path"
    cmake_args+=("-DCMAKE_C_COMPILER=$c_compiler_path")
fi

if [ "$cxx_compiler_path" != "" ]; then
    echo "INFO: C++ compiler: $cxx_compiler_path"
    cmake_args+=("-DCMAKE_CXX_COMPILER=$cxx_compiler_path")
fi

if [ "$enable_runtime" = "ON" ]; then
    cmake_args+=("-DTTMLIR_ENABLE_RUNTIME=ON")
else
    cmake_args+=("-DTTMLIR_ENABLE_RUNTIME=OFF")
fi

if [ "$enable_runtime_tests" = "ON" ]; then
    cmake_args+=("-DTTMLIR_ENABLE_RUNTIME_TESTS=ON")
else
    cmake_args+=("-DTTMLIR_ENABLE_RUNTIME_TESTS=OFF")
fi

if [ "$enable_profiler" = "ON" ]; then
    cmake_args+=("-DTT_RUNTIME_ENABLE_PERF_TRACE=ON")
fi

if [ "$enable_op_model" = "ON" ]; then
    cmake_args+=("-DTTMLIR_ENABLE_OPMODEL=ON")
fi

if [ "$enable_ccache" = "ON" ]; then
    cmake_args+=("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")
fi

if [ "$disable_python_bindings" = "ON" ]; then
    cmake_args+=("-DTTMLIR_ENABLE_BINDINGS_PYTHON=OFF")
fi

echo "INFO: Configuring Project"
echo "INFO: Running: cmake "${cmake_args[@]}""
cmake "${cmake_args[@]}"

echo "INFO: Building Project"
cmake --build $build_dir

if [ "$skip_tests" != "ON" ]; then
    echo "INFO: Running tests"
    cmake --build $build_dir -- check-ttmlir
fi

if [ "$skip_ttrt" != "ON" ]; then
    echo "INFO: Building ttrt"
    if [ -e "$build_dir"/runtime/tools/python/build/.installed ]; then
        if [ ! -e "$TTMLIR_VENV_DIR"/bin/ttrt ]; then
            echo "INFO: Did not find ttrt binary, rebuilding"
            rm -f "$build_dir"/runtime/tools/python/build/.installed
        fi
    fi
    cmake --build $build_dir -- ttrt
fi
