add_mlir_dialect_library(MLIREMITPYDialect
        EmitPyDialect.cpp
        EmitPyOps.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIREmitPyOpsIncGen
        MLIRTTOpsIncGen
        )
