file(REMOVE_RECURSE
  "CMakeFiles/MLIRTTIROpsIncGen"
  "TTIRDialect.md"
  "TTIROp.md"
  "TTIROps.cpp.inc"
  "TTIROps.h.inc"
  "TTIROpsAttrs.cpp.inc"
  "TTIROpsAttrs.h.inc"
  "TTIROpsDialect.cpp.inc"
  "TTIROpsDialect.h.inc"
  "TTIROpsEnums.cpp.inc"
  "TTIROpsEnums.h.inc"
  "TTIROpsInterfaces.cpp.inc"
  "TTIROpsInterfaces.h.inc"
  "TTIROpsTypes.cpp.inc"
  "TTIROpsTypes.h.inc"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/MLIRTTIROpsIncGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
