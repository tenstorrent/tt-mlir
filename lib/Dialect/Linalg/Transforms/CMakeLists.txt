add_mlir_library(MLIRLinalgSynchronizableOpInterfaceImpl
        SynchronizableOpInterfaceImpl.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        LINK_LIBS PUBLIC
        MLIRLinalgDialect
        MLIRD2MDialect
        MLIRIR
        )
