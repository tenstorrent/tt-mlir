add_mlir_dialect_library(MLIRTTNNDialect
        TTNNOpsAttrs.cpp
        TTNNDialect.cpp
        TTNNOps.cpp
        TTNNOpModelInterface.cpp
        TTNNOpsTypes.cpp
        TTNNWorkaroundInterface.cpp
        TTNNWorkaroundsPass.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTCoreOpsIncGen
        MLIRTTNNWorkaroundInterfaceIncGen

        LINK_LIBS PUBLIC
        TTMLIRTTNNUtils
        TTNNOpModelLib
        MLIRSCFToEmitC
        MLIRLinalgDialect
        MLIRBufferizationDialect
        MLIRMLProgramDialect
        )
