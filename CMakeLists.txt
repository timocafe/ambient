cmake_minimum_required (VERSION 3.0)

########################################################################
#
# Project information
#
########################################################################

project (ambient)

SET(CMAKE_CXX_STANDARD 20)

if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
  set(CMAKE_INSTALL_PREFIX "/opt/ambient" CACHE PATH "Ambient install prefix" FORCE)
endif(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "Type of build" FORCE)
endif(NOT CMAKE_BUILD_TYPE)
message(STATUS "Configuration type: " ${CMAKE_BUILD_TYPE})

set(AMBIENT_SOURCES_DIR "${PROJECT_SOURCE_DIR}/src")

########################################################################
#
# Options
#
########################################################################

set(AMBIENT_THREADING "none" CACHE STRINGS "cilk, openmp, none or auto.")
set(AMBIENT_MPI "disable" CACHE STRINGS "disable, single, funneled, serialized, multiple or auto.")
set(MKL_LIB_DIR "" CACHE PATH "Location of BLAS/LAPACK libraries.")
set(MKL_LIBRARIES "" CACHE STRINGS "BLAS/LAPACK libraries")
set(PLASMA_LIB_DIR "" CACHE PATH "Location of PLASMA libraries.")
set(PLASMA_LIBRARIES "" CACHE STRINGS "PLASMA libraries")
option(AMBIENT_REGRESSION_TESTS "Build regression tests" OFF)

########################################################################
#
# Configuration
#
########################################################################

set(AMBIENT_INCLUDE_DIRS)
set(AMBIENT_LIBRARIES)
set(AMBIENT_LIBRARY_DIRS)
set(AMBIENT_COMPILER_FLAGS)
set(AMBIENT_DEFINITIONS)

if(${AMBIENT_THREADING} MATCHES "cilk")
  list(APPEND AMBIENT_DEFINITIONS "-DAMBIENT_THREADING=CILK")
elseif(${AMBIENT_THREADING} MATCHES "openmp")
  find_package(OpenMP REQUIRED)
  list(APPEND AMBIENT_COMPILER_FLAGS ${OpenMP_CXX_FLAGS})
  list(APPEND AMBIENT_LIBRARIES ${OpenMP_CXX_FLAGS})
  list(APPEND AMBIENT_DEFINITIONS "-DAMBIENT_THREADING=OPENMP")
elseif(${AMBIENT_THREADING} MATCHES "none")
  list(APPEND AMBIENT_DEFINITIONS "-DAMBIENT_THREADING=NONE")
elseif(NOT ${AMBIENT_THREADING} MATCHES "auto")
  message(FATAL_ERROR "AMBIENT_THREADING has to be one of 'cilk', 'openmp', 'none' or 'auto'")
endif()
message(STATUS "Ambient threading: ${AMBIENT_THREADING}")

if(NOT MPI_CXX_FOUND AND NOT ${AMBIENT_MPI} MATCHES "disable")
  find_package(MPI)
endif(NOT MPI_CXX_FOUND AND NOT ${AMBIENT_MPI} MATCHES "disable")
if(MPI_CXX_FOUND AND NOT ${AMBIENT_MPI} MATCHES "disable")
  message(STATUS "MPI include: ${MPI_CXX_INCLUDE_PATH}")
  message(STATUS "MPI libs: ${MPI_CXX_LIBRARIES}")
  list(APPEND AMBIENT_INCLUDE_DIRS ${MPI_CXX_INCLUDE_PATH})
  list(APPEND AMBIENT_LIBRARIES ${MPI_CXX_LIBRARIES})
else(MPI_CXX_FOUND AND NOT ${AMBIENT_MPI} MATCHES "disable")
  message(STATUS "Warning: MPI is disabled")
  set(AMBIENT_MPI "disable")
endif(MPI_CXX_FOUND AND NOT ${AMBIENT_MPI} MATCHES "disable")

if(${AMBIENT_MPI} MATCHES "disable")
  list(APPEND AMBIENT_DEFINITIONS "-DAMBIENT_MPI=MPI_DISABLE")
elseif(${AMBIENT_MPI} MATCHES "single")
  list(APPEND AMBIENT_DEFINITIONS "-DAMBIENT_MPI=MPI_THREAD_SINGLE")
elseif(${AMBIENT_MPI} MATCHES "funneled")
  list(APPEND AMBIENT_DEFINITIONS "-DAMBIENT_MPI=MPI_THREAD_FUNNELED")
elseif(${AMBIENT_MPI} MATCHES "serialized")
  list(APPEND AMBIENT_DEFINITIONS "-DAMBIENT_MPI=MPI_THREAD_SERIALIZED")
elseif(${AMBIENT_MPI} MATCHES "multiple")
  list(APPEND AMBIENT_DEFINITIONS "-DAMBIENT_MPI=MPI_THREAD_MULTIPLE")
elseif(NOT ${AMBIENT_MPI} MATCHES "auto")
  message(FATAL_ERROR "AMBIENT_MPI has to be one of 'disable', 'single', 'funneled', 'serialized', 'multiple' or 'auto'")
endif()
message(STATUS "Ambient MPI: ${AMBIENT_MPI}")

find_package(Eigen3 3.3 REQUIRED NO_MODULE)


message(STATUS "Math lib dirs: ${MKL_LIB_DIR} ${PLASMA_LIB_DIR}")
message(STATUS "Math libs: ${MKL_LIBRARIES} ${PLASMA_LIBRARIES}")
list(APPEND AMBIENT_LIBRARY_DIRS ${MKL_LIB_DIR} ${PLASMA_LIB_DIR})
list(APPEND AMBIENT_LIBRARIES ${MKL_LIBRARIES} ${PLASMA_LIBRARIES})
foreach(LIB_DIR ${AMBIENT_LIBRARY_DIRS})
  list(INSERT AMBIENT_LIBRARIES 0 "-L${LIB_DIR}")
endforeach(LIB_DIR)

message(STATUS "AMBIENT DEFINITIONS: ${AMBIENT_DEFINITIONS}")
message(STATUS "AMBIENT COMPILER_FLAGS: ${AMBIENT_COMPILER_FLAGS}")
message(STATUS "AMBIENT LIBRARIES: ${AMBIENT_LIBRARIES}")
message(STATUS "AMBIENT INCLUDE DIRS: ${AMBIENT_INCLUDE_DIRS}")

########################################################################
#
# Boost for deep learning needs png and jpeg
#
########################################################################

find_package(Boost)
include_directories(SYSTEM ${Boost_INCLUDE_DIR})
include_directories(${PNG_INCLUDE_DIR})
include_directories(${JPEG_INCLUDE_DIR})
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
add_subdirectory(deep_learning)

########################################################################
#
# Ambient interface library
#
########################################################################

add_library(ambient INTERFACE)
target_include_directories(ambient
  INTERFACE
    $<BUILD_INTERFACE:${AMBIENT_SOURCES_DIR}>
    $<INSTALL_INTERFACE:include>
  INTERFACE ${AMBIENT_INCLUDE_DIRS}
)
target_compile_definitions(ambient
  INTERFACE ${AMBIENT_DEFINITIONS}
)
target_compile_options(ambient
  INTERFACE ${AMBIENT_COMPILER_FLAGS}
)
target_link_libraries(ambient
  INTERFACE ${AMBIENT_LIBRARIES}
)

add_subdirectory(examples)

########################################################################
#
# Tests
#
########################################################################

if(AMBIENT_REGRESSION_TESTS)
  include(CTest)
  enable_testing()
  add_subdirectory(tests/functional)
  add_subdirectory(tests/performance)
elseif(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  message("\n\n Ambient is a header-only library (so go ahead and 'make install') \n\n")
endif(AMBIENT_REGRESSION_TESTS)

########################################################################
#
# Configuration dump
#
########################################################################

list(APPEND AMBIENT_CONFIG_INCLUDE_DIRS ${CMAKE_INSTALL_PREFIX}/include ${AMBIENT_INCLUDE_DIRS})
list(APPEND AMBIENT_CONFIG_LIBRARY_DIRS ${CMAKE_INSTALL_PREFIX}/lib ${AMBIENT_LIBRARY_DIRS})
list(APPEND AMBIENT_CONFIG_LIBRARIES ${AMBIENT_LIBRARIES})
list(APPEND AMBIENT_CONFIG_DEFINITIONS ${AMBIENT_DEFINITIONS})
list(APPEND AMBIENT_CONFIG_COMPILE_FLAGS ${AMBIENT_COMPILER_FLAGS})

configure_file(AmbientConfig.cmake.in ${PROJECT_BINARY_DIR}/AmbientConfig.cmake)

########################################################################
#
# Install
#
########################################################################

install(DIRECTORY src/ambient src/utils
        DESTINATION include
        COMPONENT headers
        PATTERN .git EXCLUDE
        PATTERN CMakeLists.txt EXCLUDE)

install(DIRECTORY examples
        DESTINATION share
        PATTERN .git EXCLUDE
        PATTERN CMakeLists.txt EXCLUDE)

install(FILES ${PROJECT_BINARY_DIR}/AmbientConfig.cmake
        DESTINATION share)
