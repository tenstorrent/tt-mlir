file(REMOVE_RECURSE
  "../../../libMLIRTTIRPipelines.a"
  "../../../libMLIRTTIRPipelines.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRTTIRPipelines.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
