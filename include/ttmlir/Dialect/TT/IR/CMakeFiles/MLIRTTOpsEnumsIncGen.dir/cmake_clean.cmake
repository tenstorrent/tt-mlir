file(REMOVE_RECURSE
  "CMakeFiles/MLIRTTOpsEnumsIncGen"
  "TTAttr.md"
  "TTDialect.md"
  "TTOp.md"
  "TTOps.cpp.inc"
  "TTOps.h.inc"
  "TTOpsAttrDefs.cpp.inc"
  "TTOpsAttrDefs.h.inc"
  "TTOpsDialect.cpp.inc"
  "TTOpsDialect.h.inc"
  "TTOpsEnums.cpp.inc"
  "TTOpsEnums.h.inc"
  "TTOpsTypes.cpp.inc"
  "TTOpsTypes.h.inc"
  "TTType.md"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/MLIRTTOpsEnumsIncGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
