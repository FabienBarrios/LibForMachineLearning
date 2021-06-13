cmake_minimum_required(VERSION 3.19)
project(V2)

set(CMAKE_CXX_STANDARD 20)

add_library(V2 SHARED library.cpp CalculMatriciel.cpp CalculMatriciel.h PMC.cpp PMC.h)

set(EIGEN_DIR "D:/Projet/Eigen")
if(NOT EXISTS ${EIGEN_DIR})
    message(FATAL_ERROR "Please check that the set Eigen directory is valid!")
endif()
include_directories(${EIGEN_DIR})