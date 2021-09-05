set(examples vector partitioned_vector  diffusion) 

foreach(example ${examples})
    add_executable(ex_${example} ${example}.cpp)
    target_link_libraries(ex_${example}  PUBLIC OpenMP::OpenMP_CXX Eigen3::Eigen ambient)
endforeach(example)
