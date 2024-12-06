add_mlir_dialect_library(MLIRTTNNDialect
        TTNNOpsAttrs.cpp
        TTNNDialect.cpp
        TTNNOps.cpp
        TTNNOpModelInterface.cpp
        TTNNOpsTypes.cpp
        TTNNWorkaroundInterface.cpp
        TTNNWorkarounds.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTNNOpsIncGen
        MLIRTTOpsIncGen
        MLIRTTNNWorkaroundInterfaceIncGen
        TTNNOpModelLib

        LINK_LIBS PUBLIC
        TTMLIRTTNNUtils
        MLIRSCFToEmitC
        MLIRLinalgDialect
        MLIRMLProgramDialect
        TTNNOpModelLib
        )
