#pragma once

#include <vector>
#include <string>
#include "common_header.hpp"
#include "xtensor-io/ximage.hpp"

template <typename T>
xt::xarray<T> load_images(const std::vector<std::string>& paths) {
  xt::xarray<T> result {};
  bool not_reshaped = true;
  for (auto& path : paths) {
    xt::xarray<T> image = xt::transpose(load_image(path), {2, 0, 1});
    auto shape = image.shape();
    image.reshape({1, shape[0], shape[1], shape[2]});
    if (not_reshaped) {
      result.reshape({0, shape[0], shape[1], shape[2]});
      not_reshaped = false;
    }
    result = xt::concatenate(xt::xtuple(result, image));
  }
  return result;
}
