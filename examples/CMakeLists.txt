set(examples mp_gemm_log mp_gemm_simple future  matrix vector partitioned_vector atomic diffusion groups mp_gemm) 

foreach(example ${examples})
    add_executable(ex_${example} ${example}.cpp)
    target_link_libraries(ex_${example}  PUBLIC OpenMP::OpenMP_CXX Eigen3::Eigen ambient)
endforeach(example)
