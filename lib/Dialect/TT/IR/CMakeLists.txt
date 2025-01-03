add_mlir_dialect_library(TTMLIRTTDialect
        TTOpsTypes.cpp
        TTDialect.cpp
        TTOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTOpsIncGen
        )
