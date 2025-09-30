add_mlir_dialect_library(MLIRD2MDialect
        D2MDialect.cpp
        D2MOps.cpp
        D2MOpsTypes.cpp
        D2MGenericRegionOps.cpp
        D2MOpsInterfaces.cpp
        D2MTraits.cpp

        ADDITIONAL_HEADER_DIRS
        ${PROJECT_SOURCE_DIR}/include/ttmlir

        DEPENDS
        MLIRD2MOpsIncGen
        MLIRD2MOpsTypesIncGen
        MLIRD2MOpsEnumsIncGen
        MLIRD2MOpsAttrsIncGen
        D2MGenericRegionOpsIncGen
        D2MOpsInterfacesIncGen

        LINK_LIBS PUBLIC
        MLIRTTCoreDialect
        MLIRIR
        MLIRD2MUtils
        )
