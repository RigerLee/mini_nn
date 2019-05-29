#pragma once

#include "common_header.hpp"
#include "utils.hpp"

enum LAYER_TYPE { CONV, LINEAR, POOL, ACT };

template <typename T>
class Network;

template <typename T>
class Layer {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Layer() = default;
  virtual ~Layer() = default;

  virtual Matrix forward(const Matrix& in) = 0;
  virtual Matrix backward(const Matrix& dout) = 0;
  virtual Shape weight_shape() { return W_.shape(); };
  virtual Shape bias_shape() { return b_.shape(); };
  virtual void set_weight(Matrix W) { W_ = W; };
  virtual void set_bias(Matrix b) { b_ = b; };
  virtual void set_network(Network<T>* net) { net_ = net; };
  virtual LAYER_TYPE get_type() { return layer_type_; };
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