file(REMOVE_RECURSE
  "../../../libMLIRTTNNPipelines.a"
  "../../../libMLIRTTNNPipelines.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRTTNNPipelines.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
