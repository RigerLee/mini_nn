/**
 * @file test_net.cpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief to test net function whether works out or not
 * @version 1.6.0
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#include "network.hpp"
using namespace std;

int main(int argc, char** argv) {
  Network<double> net;
  auto input = xt::linspace<double>(-0.5, 0.5, 18).reshape({2, 1, 3, 3});
  xt::xarray<double> target = {2, 3};

  net.set_optimizer(new SGD<double>(0.5));

  net << new Conv2d<double>(1, 1, 3, 1, 1) << new ReLU<double>()
      << new MaxPool2d<double>(2, 2, 2) << new Linear<double>(9, 5)
      << new Loss<double>(CROSS_ENTROPY);
  // overfit
  for (int i = 0; i < 100; ++i) {
    std::cout << net.forward(input, target) << std::endl;
    net.backward();
  }

  return 0;
}