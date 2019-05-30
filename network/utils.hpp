/**
 * @file utils.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief 
 * @version 1.6.0
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once

#include <common_header.hpp>
/**
 * @brief gets the shape of the input array
 * 
 * @tparam T 
 * @param matrix 
 */
template <typename T>
void cout_shape(const xt::xarray<T>& matrix) {
  std::cout << "shape: [";
  for (size_t i = 0; i < matrix.shape().size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << matrix.shape(i);
  }
  std::cout << "]\n";
}