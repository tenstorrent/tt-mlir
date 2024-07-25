add_mlir_dialect_library(MLIRTTNNTransforms
        Passes.cpp
        TTNNToCpp.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTNNPassesIncGen
        MLIRTTOpsIncGen
        )

target_include_directories(MLIRTTNNTransforms PUBLIC ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common)
