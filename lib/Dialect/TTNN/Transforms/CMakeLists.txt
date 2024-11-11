add_mlir_dialect_library(MLIRTTNNTransforms
        TTNNLayout.cpp
        Passes.cpp
        Optimizer.cpp
        TTNNToCpp.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTNNPassesIncGen
        MLIRTTOpsIncGen

        LINK_LIBS PUBLIC
        MLIRTTNNDialect
        )

target_include_directories(MLIRTTNNTransforms PUBLIC ${PROJECT_BINARY_DIR}/include/ttmlir/Target/Common)
