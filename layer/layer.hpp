#pragma once

#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "xtensor/xpad.hpp"
#include "optimizer.hpp"

template <typename T>
class Layer {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Layer() = default;
  virtual ~Layer() = default;

  virtual Matrix forward(const Matrix& in) {return Matrix();};
  virtual Matrix backward(const Matrix& dout) {return Matrix();};
  virtual void init_weight() {};
  virtual void init_bias() {};

protected:
  Shape in_shape_;
  Shape out_shape_;
  Matrix in_;
  Matrix din_;
  Matrix W_;
  Matrix dW_;
  Matrix b_;
  Matrix db_;
  Optimizer<T> optimizer_;

};