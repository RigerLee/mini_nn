add_executable(
  Test_Layer
  test_layer.cpp
)

add_executable(
        Test_Net
        test_net.cpp
)

add_executable(
        Test_Loader
        test_loader.cpp
)

add_executable(
        Test_MNIST
        test_mnist.cpp
)

target_link_libraries(Test_Layer ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
target_link_libraries(Test_Net ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
target_link_libraries(Test_Loader ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OIIO})
target_link_libraries(Test_MNIST ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES} ${OIIO})
