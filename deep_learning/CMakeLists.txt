set(examples logistic_regression) 


foreach(example ${examples})
    add_executable(ex_${example} ${example}.cpp)
    if(${AMBIENT_THREADING} MATCHES "openmp")
        target_link_libraries(ex_${example} PUBLIC OpenMP::OpenMP_CXX Eigen3::Eigen ambient ${PNG_LIBRARY_RELEASE} ${JPEG_LIBRARY_RELEASE})
    else()
        target_link_libraries(ex_${example} PUBLIC Eigen3::Eigen ambient ${PNG_LIBRARY_RELEASE} ${JPEG_LIBRARY_RELEASE} )
    endif()
endforeach(example)
