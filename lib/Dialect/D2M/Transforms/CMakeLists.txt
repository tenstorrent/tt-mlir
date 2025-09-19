add_mlir_dialect_library(MLIRD2MTransforms
  Allocate.cpp
  LowerToLayout.cpp
  GenericApplyInterchange.cpp
  GenericTileComputeLoops.cpp
  InsertDstRegisterAccess.cpp
  GenericLinearizeMemref.cpp
  GenericGenerateDatamovement.cpp
  GenericLowerDMAs.cpp
  GenericHWThreadSelection.cpp
  GenericGenerateLoops.cpp
  GenericRegionsToFuncs.cpp

  ADDITIONAL_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}/include/ttmlir

  DEPENDS
  MLIRD2MPassesIncGen

  LINK_LIBS PUBLIC
  MLIRD2MDialect
  MLIRD2MUtils
  MLIRD2MAnalysis
  MLIRTTCoreDialect
  MLIRPass
  MLIRTransforms
)
