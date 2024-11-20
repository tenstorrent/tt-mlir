file(REMOVE_RECURSE
  "../../../libMLIRTTMetalPipelines.a"
  "../../../libMLIRTTMetalPipelines.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRTTMetalPipelines.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
