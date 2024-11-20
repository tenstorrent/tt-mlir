file(REMOVE_RECURSE
  "../../../libMLIRTTKernelDialect.a"
  "../../../libMLIRTTKernelDialect.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRTTKernelDialect.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
