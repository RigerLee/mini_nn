/**
 * @file layer_base.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief the attribute of the base of the layter
 * @version 1.6.0
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 *
 */

#pragma once

#include "common_header.hpp"
#include "utils.hpp"
/**
 * @brief enumerate the layer type
 *
 */
enum LAYER_TYPE { CONV, LINEAR, POOL, ACT };

/**
 * @brief the class of network
 *
 */
template <typename T>
class Network;

/**
 * @brief the class of the layer
 *
 */
template <typename T>
class Layer {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Layer() = default;
  virtual ~Layer() = default;
  /**
   * @brief forward function in the network
   *
   * @param in    the input matrix
   * @return      the output matrix
   */
  virtual Matrix forward(const Matrix& in) = 0;
  /**
   * @brief backward function in the networ
   *
   * @param dout    the derivative of output matrix
   * @return        the derivative of input matrix
   */
  virtual Matrix backward(const Matrix& dout) = 0;
  /**
   * @brief
   *
   * @return the shape of weight matrix
   */
  virtual Shape weight_shape() { return W_.shape(); };
  /**
   * @brief
   *
   * @return the shape of bias matrix
   */
  virtual Shape bias_shape() { return b_.shape(); };
  /**
   * @brief Set the weight object
   *
   * @param W   the weight matrix
   */
  virtual void set_weight(const Matrix& W) { W_ = W; };
  /**
   * @brief Set the bias object
   *
   * @param b   the bias matrix
   */
  virtual void set_bias(const Matrix& b) { b_ = b; };
  /**
   * @brief Set the network object
   *
   * @param net    the network object
   */
  virtual void set_network(Network<T>* net) { net_ = net; };
  /**
   * @brief Get the type of the layer
   *
   * @return the type of the layer
   */
  virtual LAYER_TYPE get_type() { return layer_type_; };
  /**
   * @brief Get the fan
   *
   * @return the fan
   */
  virtual size_t get_fan() { return 0; };

protected:
  LAYER_TYPE layer_type_;
  Matrix in_;
  Matrix din_;
  Matrix W_;
  Matrix dW_;
  Matrix b_;
  Matrix db_;
  Network<T>* net_;
};