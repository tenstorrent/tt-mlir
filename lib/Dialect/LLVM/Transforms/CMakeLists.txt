add_mlir_dialect_library(MLIRLLVMTransforms
        EmitWrapperFuncs.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTOpsIncGen
        )

target_link_libraries(MLIRLLVMTransforms PUBLIC coverage_config)
