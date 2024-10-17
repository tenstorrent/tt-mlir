set(LLVM_TARGET_DEFINITIONS Decompose.td)
mlir_tablegen(Decompose.inc -gen-rewriters)
add_public_tablegen_target(MLIRTTIRDecomposeIncGen)

add_mlir_dialect_library(MLIRTTIRTransforms
        Allocate.cpp
        Constant.cpp
        Generic.cpp
        Layout.cpp
        Decompose.cpp
        Optimizer.cpp
        Transforms.cpp
        Utility.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRDecomposeIncGen
        MLIRTTIRPassesIncGen
        MLIRTTOpsIncGen
        )
