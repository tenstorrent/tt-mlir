file(REMOVE_RECURSE
  "../../../libMLIRTTNNAnalysis.a"
  "../../../libMLIRTTNNAnalysis.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRTTNNAnalysis.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
