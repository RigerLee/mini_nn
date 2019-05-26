# add_executable(
#   Test_Layer
#   test_layer.cpp
# )
#
# target_link_libraries(Test_Layer ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})

add_executable(
  Test_Net
  test_net.cpp
)

target_link_libraries(Test_Net ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})