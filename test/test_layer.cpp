#include <iostream>

#include "layer_common.hpp"

int main( int argc, char** argv ) {

  Conv2d<double> test(1, 1);
  auto a = xt::ones<double>({2, 1, 5 ,5});
  // seems work
  auto temp = test.forward(a);
  auto temp_shape_size = temp.shape().size();
  for (size_t i = 0; i < temp_shape_size; ++i) {
    std::cout << temp.shape(i) << ", ";
  }
  std::cout << std::endl;
  std::cout << temp << std::endl;
  std::cout << "---------------------------------------" << std::endl;
  // not sure
  auto temp1 = test.backward(temp);
  auto temp1_shape_size = temp1.shape().size();
  for (size_t i = 0; i < temp1_shape_size; ++i) {
    std::cout << temp1.shape(i) << ", ";
  }
  std::cout << std::endl;
  std::cout << temp1 << std::endl;
  return 0;
}