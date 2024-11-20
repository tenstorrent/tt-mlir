file(REMOVE_RECURSE
  "CMakeFiles/MLIRTTKernelOpsAttributesIncGen"
  "TTKernelAttr.md"
  "TTKernelAttrInterfaces.cpp.inc"
  "TTKernelAttrInterfaces.h.inc"
  "TTKernelDialect.md"
  "TTKernelOp.md"
  "TTKernelOps.cpp.inc"
  "TTKernelOps.h.inc"
  "TTKernelOpsAttrDefs.cpp.inc"
  "TTKernelOpsAttrDefs.h.inc"
  "TTKernelOpsDialect.cpp.inc"
  "TTKernelOpsDialect.h.inc"
  "TTKernelOpsEnums.cpp.inc"
  "TTKernelOpsEnums.h.inc"
  "TTKernelOpsTypes.cpp.inc"
  "TTKernelOpsTypes.h.inc"
  "TTKernelType.md"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/MLIRTTKernelOpsAttributesIncGen.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
