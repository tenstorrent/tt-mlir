add_mlir_dialect_library(MLIRTTCoreDialect
        TTCoreDialect.cpp
        TTCoreOps.cpp
        TTCoreOpsTypes.cpp
        Utils.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTCoreOpsIncGen

        LINK_LIBS PUBLIC
        MLIRD2MUtils
        MLIRMemRefDialect
        MLIRQuantDialect
        )
