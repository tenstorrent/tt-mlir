add_mlir_dialect_library(MLIRTT
        TTTypes.cpp
        TTDialect.cpp
        TTOps.cpp
        TTPasses.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRTTOpsIncGen
        MLIRTTPassesIncGen

        LINK_LIBS PUBLIC
        MLIRIR
        MLIRInferTypeOpInterface
        MLIRFuncDialect
        )
