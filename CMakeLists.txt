cmake_minimum_required(VERSION 3.6)
project(mi-prc)

# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

set(SOURCE_FILES main.cpp)
add_executable(openmp_test_clion ${SOURCE_FILES})