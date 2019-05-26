#pragma once

#include <common_header.hpp>

template <typename T>
void cout_shape(const xt::xarray<T>& matrix) {
  std::cout << "shape: [";
  for (size_t i = 0; i < matrix.shape().size(); ++i) {
    if (i != 0) std::cout << ", ";
    std::cout << matrix.shape(i);
  }
  std::cout << "]\n";
}