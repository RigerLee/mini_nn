/**
 * @file init.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief Init the network. Containing two funcitons: kaiming_normal and kaiming_uniform
 * @version 1.6.0
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once

#include "convolution.hpp"
#include "linear.hpp"

/**
 * @brief kaiming normal distribution
 *
 * @tparam T
 * @param layer: the layer of the network
 * @param mode: how to convolution
 * @details
 * According to the method described by He, K et al. in 'Delving deep into
 * rectifiers: Surpassing human-level performance on ImageNet classification' in
 * 2015, the input tensor or variable is filled with a uniform distribution
 * generation value. The resulting value in the tensor is sampled from U(-bound,
 * bound), where bound = sqrt(2/((1 + a^2) * fan_in)) * sqrt(3). Also known as
 * He initialisation.
 */
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

/**
 * @brief kaiming uniform distribution
 *
 * @tparam T
 * @param layer: the layer of the network
 * @param mode: how to convolution
 * @details
 * According to the method described by He, K et al. in 'Delving deep into
 * rectifiers: Surpassing human-level performance on ImageNet classification' in
 * 2015, the input tensor or variable is filled with a uniform distribution
 * generation value. The resulting value in the tensor is sampled from U(-bound,
 * bound), where bound = sqrt(2/((1 + a^2) * fan_in)) * sqrt(3). Also known as
 * He initialisation.
 */
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