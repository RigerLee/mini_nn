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

add_executable(
        Test_Loader
        test_loader.cpp
)

target_link_libraries(Test_Net ${BLAS_LIBRARIES} ${LAPACK_LIBRARIES})
target_link_libraries(Test_Loader ${OIIO})
