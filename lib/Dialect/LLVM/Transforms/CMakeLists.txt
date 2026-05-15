add_mlir_dialect_library(MLIRLLVMTransforms
        PromoteToStaticAllocation.cpp
        EmitWrapperFuncs.cpp
        NonContiguousMemrefCopyToLinalg.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTIROpsIncGen
        MLIRTTIRPassesIncGen
        MLIRTTCoreOpsIncGen
        )
