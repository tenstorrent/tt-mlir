file(REMOVE_RECURSE
  "../../../libMLIRTTNNTransforms.a"
  "../../../libMLIRTTNNTransforms.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRTTNNTransforms.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
