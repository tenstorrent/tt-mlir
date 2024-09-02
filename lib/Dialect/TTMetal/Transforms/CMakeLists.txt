add_mlir_dialect_library(MLIRTTMetalTransforms
        Passes.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTMetalOpsIncGen
        MLIRTTMetalPassesIncGen
        MLIRTTOpsIncGen
        )

target_include_directories(MLIRTTMetalTransforms PUBLIC ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common)
