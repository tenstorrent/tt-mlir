file(REMOVE_RECURSE
  "../libMLIRScheduler.a"
  "../libMLIRScheduler.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang CXX)
  include(CMakeFiles/MLIRScheduler.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
