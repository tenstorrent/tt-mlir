include(ExternalProject)

set(TORCH_MLIR_VERSION "b1a232222f12d4e5640fd62320d02f6c832bdc4e")

ExternalProject_Add(
    torch-mlir
    PREFIX ${TTMLIR_TOOLCHAIN_DIR}
    CMAKE_GENERATOR Ninja
    CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=Release
    -DPython3_FIND_VIRTUALENV=ONLY
    -DCMAKE_INSTALL_PREFIX=${TTMLIR_TOOLCHAIN_DIR}
    -DMLIR_DIR=${TTMLIR_TOOLCHAIN_DIR}/llvm-project/lib/cmake/mlir/
    -DLLVM_DIR=${TTMLIR_TOOLCHAIN_DIR}/llvm-project/lib/cmake/llvm/
    -DMLIR_ENABLE_BINDINGS_PYTHON=ON
    -DLLVM_TARGETS_TO_BUILD=host
    GIT_REPOSITORY https://github.com/llvm/torch-mlir
    GIT_TAG ${TORCH_MLIR_VERSION}
)
