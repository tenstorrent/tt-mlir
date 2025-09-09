add_mlir_dialect_library(MLIRTTNNDialect
        TTNNOpsAttrs.cpp
        TTNNDialect.cpp
        TTNNOps.cpp
        TTNNOpsTypes.cpp
        TTNNWorkaroundsPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTCoreOpsIncGen
        MLIRTTNNOpModelInterfaceIncGen
        MLIRTTNNTensorSpecInterfaceIncGen
        MLIRTTNNWorkaroundInterfaceIncGen
        MLIRTTNNKernelInterfaceIncGen

        LINK_LIBS PUBLIC
        MLIRBufferizationDialect
        MLIRLinalgDialect
        MLIRMLProgramDialect
        MLIRSCFToEmitC
        MLIRTTNNInterfaces
        TTMLIRTTNNUtils
        TTNNOpModelLib
        )
