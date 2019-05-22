add_executable(
  Test
  test_layer.cpp
)

target_link_libraries(Test ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})