file(REMOVE_RECURSE
  "../../../libMLIRTTIRTransforms.a"
  "../../../libMLIRTTIRTransforms.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRTTIRTransforms.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
