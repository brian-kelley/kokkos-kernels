cmake_minimum_required(VERSION 3.16 FATAL_ERROR)
project(lapis_package)
find_package(Kokkos REQUIRED)

add_executable(spmv main.cpp)
target_link_libraries(spmv Kokkos::kokkos)
