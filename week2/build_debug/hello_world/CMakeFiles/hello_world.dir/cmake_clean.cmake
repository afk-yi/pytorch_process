file(REMOVE_RECURSE
  "../bin/hello_world"
  "../bin/hello_world.pdb"
  "CMakeFiles/hello_world.dir/hello_world.cu.o"
  "CMakeFiles/hello_world.dir/hello_world.cu.o.d"
)

# Per-language clean rules from dependency scanning.
foreach(lang CUDA)
  include(CMakeFiles/hello_world.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
