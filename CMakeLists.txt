cmake_minimum_required(VERSION 3.18)
project(lazyipdca LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options(-O3 -march=native -fno-math-errno)

option(LAZYIPDCA_PYTHON "Build the pybind11 Python module" OFF)
option(LAZYIPDCA_CUTEST "Build the CUTEst benchmark runner" OFF)
option(LAZYIPDCA_TESTS  "Build C++ smoke tests" ON)

find_package(OpenMP REQUIRED)

# ---- MUMPS (sequential) + METIS -------------------------------------------
# Priority: conda env (CMAKE_PREFIX_PATH / CONDA_PREFIX) then system.
if(DEFINED ENV{CONDA_PREFIX})
  list(APPEND CMAKE_PREFIX_PATH $ENV{CONDA_PREFIX})
endif()
find_path(MUMPS_INCLUDE_DIR dmumps_c.h
          PATH_SUFFIXES include mumps_seq include/mumps_seq)
find_library(DMUMPS_LIB NAMES dmumps_seq dmumps)
find_library(MUMPS_COMMON_LIB NAMES mumps_common_seq mumps_common)
find_library(METIS_LIB NAMES metis)
find_library(MPISEQ_LIB NAMES mpiseq_seq mpiseq)   # MUMPS fake-MPI (optional)
find_library(ESMUMPS_LIB NAMES esmumps)             # scotch glue (optional)
find_library(PORD_LIB NAMES pord pord_seq)          # optional
if(NOT MUMPS_INCLUDE_DIR OR NOT DMUMPS_LIB)
  message(FATAL_ERROR "MUMPS (sequential) not found. Install mumps-seq "
          "(conda: mumps-seq; apt: libmumps-seq-dev) and/or set CMAKE_PREFIX_PATH.")
endif()
message(STATUS "MUMPS include: ${MUMPS_INCLUDE_DIR}")
message(STATUS "MUMPS libs: ${DMUMPS_LIB} ${MUMPS_COMMON_LIB} ${MPISEQ_LIB}")

find_library(LAPACK_LIB NAMES openblas lapack)
find_library(BLAS_LIB NAMES openblas blas)
find_library(GFORTRAN_LIB NAMES gfortran)

add_library(lazyipdca_core
  src/mumps_solver.cpp
  src/lanczos.cpp
  third_party/qdldl/qdldl.c
  src/solver.cpp)
find_path(LAZYIPDCA_EIGEN3 Eigen/Core PATHS /usr/include/eigen3 /usr/local/include/eigen3 $ENV{CONDA_PREFIX}/include/eigen3 REQUIRED)
message(STATUS "Eigen3: ${LAZYIPDCA_EIGEN3}")
target_include_directories(lazyipdca_core PUBLIC ${LAZYIPDCA_EIGEN3} ${CMAKE_CURRENT_SOURCE_DIR}/third_party/qdldl
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  ${MUMPS_INCLUDE_DIR})
target_link_libraries(lazyipdca_core PUBLIC
  ${DMUMPS_LIB} ${MUMPS_COMMON_LIB} OpenMP::OpenMP_CXX ${LAPACK_LIB} ${BLAS_LIB})
foreach(optlib MPISEQ_LIB METIS_LIB ESMUMPS_LIB PORD_LIB GFORTRAN_LIB)
  if(${optlib})
    target_link_libraries(lazyipdca_core PUBLIC ${${optlib}})
  endif()
endforeach()

if(LAZYIPDCA_TESTS)
  add_executable(smoke tests/smoke.cpp)
  target_link_libraries(smoke PRIVATE lazyipdca_core)
  add_executable(bp_test tests/bp_test.cpp)
  target_link_libraries(bp_test PRIVATE lazyipdca_core)
  add_executable(best_feasible_test tests/best_feasible_test.cpp)
  target_link_libraries(best_feasible_test PRIVATE lazyipdca_core)
  add_executable(inertia_modes_test tests/inertia_modes_test.cpp)
  target_include_directories(inertia_modes_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_executable(inertial_acceptance_test tests/inertial_acceptance_test.cpp)
  target_link_libraries(inertial_acceptance_test PRIVATE lazyipdca_core)
  add_executable(rho_policy_test tests/rho_policy_test.cpp)
  target_include_directories(rho_policy_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  add_executable(atom_oracle_test tests/atom_oracle_test.cpp)
  target_include_directories(atom_oracle_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
endif()

if(LAZYIPDCA_PYTHON)
  find_package(pybind11 CONFIG REQUIRED)
  pybind11_add_module(_lazyipdca python/module.cpp)
  target_link_libraries(_lazyipdca PRIVATE lazyipdca_core)
  install(TARGETS _lazyipdca DESTINATION lazyipdca)
endif()

if(LAZYIPDCA_CUTEST)
  # Requires a decoded CUTEst installation; point CUTEST/MYARCH as usual.
  set(CUTEST_DIR $ENV{CUTEST} CACHE PATH "CUTEst root")
  set(CUTEST_ARCH $ENV{MYARCH} CACHE STRING "CUTEst architecture string")
  find_library(CUTEST_LIB NAMES cutest
               PATHS ${CUTEST_DIR}/objects/${CUTEST_ARCH}/double ${CUTEST_DIR}/lib)
  if(NOT CUTEST_LIB)
    message(FATAL_ERROR "libcutest not found; export CUTEST and MYARCH")
  endif()
  add_executable(lazyipdca_cutest runner/cutest_runner.cpp)
  target_include_directories(lazyipdca_cutest PRIVATE ${CUTEST_DIR}/include)
  target_link_libraries(lazyipdca_cutest PRIVATE lazyipdca_core ${CUTEST_LIB} ${GFORTRAN_LIB})
endif()
