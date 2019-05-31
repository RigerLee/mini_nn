/**
 * @file activation.hpp
 * @author RuiJian Li (lirj@shanghaitech.edu.cn), YiFan
 * Cao (caoyf@shanghaitech.edu.cn), YanPeng Hu (huyp@shanghaitech.edu.cn)
 * @brief
 * @version 1.6.0
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 *
 */
#pragma once

#include "layer_base.hpp"

/**
 * @brief ReLu Class, the rectifier is an activation function
 *
 * @tparam T
 * @details
 * In the context of artificial neural networks, the rectifier is an
activation function defined as the positive part of its argument:

\f[ f(x)=x^{+}=\max(0,x) \f]

where \f$ x \f$ is the input to a neuron. This is also known as a ramp
function and is analogous to half-wave rectification in electrical engineering.
This activation function was first introduced to a dynamical network by
Hahnloser et al. in 2000 with strong biological motivations and mathematical
justifications. It has been demonstrated for the first time in 2011 to enable
better training of deeper networks,compared to the widely-used activation
functions prior to 2011, e.g., the logistic sigmoid (which is inspired by
probability theory; see logistic regression) and its more practical counterpart,
the hyperbolic tangent. The rectifier is, as of 2017, the most popular
activation function for deep neural networks.
 */
template <typename T>
class ReLU : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  /**
   * @brief Construct a new ReLU object
   *
   */
  ReLU();

  /**
   * @brief Destroy the ReLU object
   *
   */
  virtual ~ReLU() = default;

  /**
   * @brief forward function in the network
   *
   * @tparam T
   * @param in : the input matrix
   * @return Matrix : the output matrix
   * @details
   */
  virtual Matrix forward(const Matrix& in) override;

  /**
   * @brief backward function in the network
   *
   * @param dout : the input matrix
   * @return Matrix : the output matrix
   */
  virtual Matrix backward(const Matrix& dout) override;
};

#include "activation_impl.hpp"