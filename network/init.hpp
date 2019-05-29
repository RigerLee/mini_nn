#pragma once

#include "convolution.hpp"
#include "linear.hpp"

template <typename T>
void kaiming_normal(Layer<T>& layer, std::string mode = "ReLU") {
  // Only init conv and linear
  if (layer.get_type() != CONV && layer.get_type() != LINEAR) return;
  size_t fan_in = layer.get_fan();
  // ReLU or LeakyReLU
  T a = mode == "ReLU" ? 0.0 : 0.01;
  T std = sqrt(2.0 / ((1 + pow(a, 2)) * fan_in));
  xt::xarray<T> weight = xt::random::randn(layer.weight_shape(), (T)0., std);
  xt::xarray<T> bias = xt::random::randn(layer.bias_shape(), (T)0., std);
  layer.set_weight(weight);
  layer.set_bias(bias);
}

template <typename T>
void kaiming_uniform(Layer<T>& layer, std::string mode = "ReLU") {
  if (layer.get_type() != CONV && layer.get_type() != LINEAR) return;
  size_t fan_in = layer.get_fan();
  T a = mode == "ReLU" ? 0.0 : 0.01;
  T bound = sqrt(6.0 / ((1 + pow(a, 2)) * fan_in));
  xt::xarray<T> weight = xt::random::rand(layer.weight_shape(), -bound, bound);
  xt::xarray<T> bias = xt::random::rand(layer.bias_shape(), -bound, bound);
  layer.set_weight(weight);
  layer.set_bias(bias);
}