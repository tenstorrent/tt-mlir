add_mlir_dialect_library(MLIRTTNNTransforms
        Optimizer.cpp
        Passes.cpp
        TTNNLayout.cpp
        TTNNDecomposeLayouts.cpp
        TTNNToCpp.cpp
        Workarounds/Decomposition/ReduceOpsRewritePattern.cpp
        Workarounds/Decomposition/RepeatOpRewritePattern.cpp
	Workarounds/TTNNWorkarounds.cpp

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
