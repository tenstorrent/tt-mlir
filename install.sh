#!/bin/bash
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

# Function to display help
show_help() {
    echo "Usage: $0 [options]..."
    echo "  -h, --help                       Show this help message."
    echo "  --clean                          Remove build workspaces."

    echo "  --debug                          Set the build type as Debug."
    echo "  --release                        Set the build type as Release."

    echo "  --opmodel                        Op Model work settings."
    echo "  --d2m                            D2M work settings."
    echo "  --explorer                       Explorer work settings."

    echo "  --speedy                         Speedy CI build settings."
    echo "  --tracy                          Tracy CI build settings."

    echo "  --skip-tests                     Skip running check-ttmlir tests."
    echo "  --skip-ttrt                      Skip building ttrt utils."
}

clean() {
    echo "INFO: Removing build artifacts!"
    rm -rf build_Release* build_Debug* build_RelWithDebInfo* build_ASan* build_TSan* build built
    rm -rf third_party/tt-metal/src/tt-metal/build third_party/tt-metal/src/tt-metal-stamp
}


# options and defaults:
build_type="Release"
build_preset=""
enable_runtime="OFF"
enable_runtime_tests="OFF"
enable_profiler="OFF"
enable_op_model="OFF"
enable_emitc="OFF"
enable_explorer="OFF"
enable_runtime_debug="OFF"
enable_pykernel="OFF"
enable_stablehlo="OFF"
skip_tests="OFF"
skip_ttrt="OFF"
# to add: TTMLIR_ENABLE_TTIRTONVVM, CODE_COVERAGE

declare -a cmake_args

OPTIONS=h
LONGOPTIONS="
help
clean
debug
release
opmodel
d2m
explorer
speedy
tracy
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
            build_type="Debug"; enable_runtime_debug="ON";;
        --release)
            build_type="Release";;
        --opmodel)
            build_preset="opmodel"; enable_runtime="ON"; enable_op_model="ON"; enable_profiler="ON";;
        --d2m)
            build_preset="d2m"; enable_runtime="ON"; enable_runtime_tests="ON"; enable_stablehlo="ON";;
        --explorer)
            build_preset="explorer"; enable_runtime="ON"; enable_explorer="ON"; enable_runtime_debug="ON";;
        --speedy)
            enable_runtime="ON"; enable_op_model="ON"; enable_emitc="ON";;
        --tracy)
            enable_runtime="ON"; enable_profiler="ON"; enable_emitc="ON"; enable_runtime_debug="ON"; enable_explorer="ON"; enable_pykernel="ON";;
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
    echo "ERROR: Invalid build type '$build_type'. Allowed values may include: Release, Debug, RelWithDebInfo, ASan, TSan."
    show_help
    exit 1
fi

# If build-dir is not specified
# Use build_type and enable_profiler setting to choose a default path
if [ -z "$build_dir" ]; then
    build_dir="build_${build_preset}_${build_type}"
    # Create and link the build directory
    mkdir -p $build_dir
    ln -nsf $build_dir build
fi


if [ -z "$TTMLIR_ENV_ACTIVATED" ]; then
    source env/activate
fi

# Debug output to verify parsed options
echo "INFO: Build type: $build_type"
echo "INFO: Build directory: $build_dir"

# Prepare cmake arguments
cmake_args+=("-B" "$build_dir")
cmake_args+=("-G" "Ninja")
cmake_args+=("-DCMAKE_BUILD_TYPE=$build_type")

cmake_args+=("-DCMAKE_C_COMPILER=clang-17")
cmake_args+=("-DCMAKE_CXX_COMPILER=clang++-17")
cmake_args+=("-DCMAKE_CXX_COMPILER_LAUNCHER=ccache")

if [ "$enable_runtime" = "ON" ]; then
    cmake_args+=("-DTTMLIR_ENABLE_RUNTIME=ON")
fi

if [ "$enable_runtime_debug" = "ON" ]; then
    cmake_args+=("-DTT_RUNTIME_DEBUG=ON")
fi

if [ "$enable_runtime_tests" = "ON" ]; then
    cmake_args+=("-DTTMLIR_ENABLE_RUNTIME_TESTS=ON")
fi

if [ "$enable_pykernel" = "ON" ]; then
    cmake_args+=("-DTTMLIR_ENABLE_PYKERNEL=ON")
fi

if [ "$enable_stablehlo" = "ON" ]; then
    cmake_args+=("-DTTMLIR_ENABLE_STABLEHLO=ON")
fi

if [ "$enable_profiler" = "ON" ]; then
    cmake_args+=("-DTT_RUNTIME_ENABLE_PERF_TRACE=ON")
fi

if [ "$enable_op_model" = "ON" ]; then
    cmake_args+=("-DTTMLIR_ENABLE_OPMODEL=ON")
fi

if [ "$enable_explorer" = "ON" ]; then
    cmake_args+=("-DTTMLIR_ENABLE_EXPLORER=ON")
fi


echo "INFO: Configuring Project"
echo "INFO: Running: cmake ${cmake_args[@]}"
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
    echo "INFO: ttrt build done, now try:"
    echo "  ttrt query --save-artifacts && export SYSTEM_DESC_PATH=$(pwd)/ttrt-artifacts/system_desc.ttsys"
fi
