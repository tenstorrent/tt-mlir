file(REMOVE_RECURSE
  "CMakeFiles/MLIRTTMetalOpsAttributesIncGen"
  "TTMetalAttr.md"
  "TTMetalDialect.md"
  "TTMetalOp.md"
  "TTMetalOps.cpp.inc"
  "TTMetalOps.h.inc"
  "TTMetalOpsAttrDefs.cpp.inc"
  "TTMetalOpsAttrDefs.h.inc"
  "TTMetalOpsDialect.cpp.inc"
  "TTMetalOpsDialect.h.inc"
  "TTMetalOpsEnums.cpp.inc"
  "TTMetalOpsEnums.h.inc"
  "TTMetalOpsTypes.cpp.inc"
  "TTMetalOpsTypes.h.inc"
  "TTMetalType.md"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/MLIRTTMetalOpsAttributesIncGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
