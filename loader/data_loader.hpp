#pragma once

#include <string>
#include <vector>
#include "xtensor-io/ximage.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"

template <typename T>
void image_normalize(xt::xarray<T>& image,
                     const xt::xarray<T>& mean,
                     const xt::xarray<T>& std_) {
  auto shape = image.shape();
  size_t C = shape[0];
  size_t H = shape[1];
  size_t W = shape[2];
  if (C != mean.size() || C != std_.size())
    throw std::runtime_error("invalid number of channels");
  for (size_t c = 0; c < C; ++c) {
    T mean_c = mean(c);
    T std_c = std_(c);
    auto begin = image.begin() + c * H * W;
    std::transform(begin, begin + H * W, begin,
                   [=](auto&& v) { return (v - mean_c) / std_c; });
  }
}

template <typename T>
xt::xarray<T> load_images(const std::vector<std::string>& paths,
                          const xt::xarray<T>& mean,
                          const xt::xarray<T>& std) {
  typename xt::xarray<T>::shape_type rshape = {0};
  xt::xarray<T> result(rshape);
  bool not_reshaped = true;
  for (auto& path : paths) {
    xt::xarray<T> image = xt::transpose(xt::load_image(path), {2, 0, 1});
    image_normalize<T>(image, mean, std);
    auto shape = image.shape();
    size_t C = shape[0];
    size_t H = shape[1];
    size_t W = shape[2];
    image.reshape({1, C, H, W});
    if (not_reshaped) {
      result.reshape({0, C, H, W});
      not_reshaped = false;
    }
    result = xt::concatenate(xt::xtuple(result, image));
  }
  return result;
}
