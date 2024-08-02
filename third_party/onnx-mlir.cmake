include(ExternalProject)

set(ONNX_MLIR_VERSION "67ea9b55f3a55cf4e7f351c56ebf54fda7bbb365")

ExternalProject_Add(
    onnx-mlir
    PREFIX ${TTMLIR_TOOLCHAIN_DIR}
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
    -DMLIR_DIR=${TTMLIR_TOOLCHAIN_DIR}/llvm-project/build/lib/cmake/mlir
    -DCMAKE_INSTALL_PREFIX=${TTMLIR_TOOLCHAIN_DIR}
    -DONNX_MLIR_BUILD_TESTS=OFF
    -DONNX_MLIR_ENABLE_JAVA=OFF
    GIT_REPOSITORY https://github.com/onnx/onnx-mlir/
    GIT_TAG ${ONNX_MLIR_VERSION}
)
