cmake_minimum_required(VERSION 3.1)
project(mini_nn)

#set(XTENSOR_DIR your_path/xtensor/build)
find_package(xtensor REQUIRED)
find_package(xtensor-blas REQUIRED)
# for xtensor-blas
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

include_directories(
  ${XTENSOR_INCLUDE_DIR}
  ${XTENSOR_BLAS_INCLUDE_DIR}
  optimizer
  layer
  loss
  loader
  network
)

add_subdirectory(test)
