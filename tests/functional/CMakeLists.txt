
add_executable( transpose   transpose.cpp )
add_executable( resize      resize.cpp    )
add_executable( rm_cols     rm_cols.cpp   )
add_executable( rm_rows     rm_rows.cpp   )
add_executable( elements    elements.cpp  )
add_executable( identity    identity.cpp  )
add_executable( trace       trace.cpp     )
add_executable( add         add.cpp       )
add_executable( sub         sub.cpp       )
add_executable( qr          qr.cpp        )
add_executable( lq          lq.cpp        )
add_executable( inverse     inverse.cpp   )
add_executable( cast        cast.cpp      )
add_executable( sqrt        sqrt.cpp      )
add_executable( conj        conj.cpp      )
add_executable( adjoint     adjoint.cpp   )
add_executable( hermitian   hermitian.cpp )

foreach( functional_test
         add
         sub
         resize
         rm_cols
         rm_rows
         transpose
         elements
         identity
         trace
         qr
         lq
         inverse
         cast
         sqrt
         conj
         adjoint
         hermitian
       )
    if(${AMBIENT_THREADING} MATCHES "openmp")
        target_link_libraries (${functional_test} PUBLIC  Eigen3::Eigen OpenMP::OpenMP_CXX ambient)
    else()
        target_link_libraries (${functional_test} PUBLIC  Eigen3::Eigen ambient)
    endif()
    add_test(${functional_test} ${functional_test})
endforeach(functional_test)
