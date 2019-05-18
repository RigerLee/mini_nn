#pragma once

#include <iostream>
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xview.hpp"
#include "optimizer.hpp"

template <typename T>
class Layer {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Layer() = default;
  virtual ~Layer() = default;

  virtual Matrix forward() {return Matrix();};
  virtual Matrix backward() {return Matrix();};
  virtual void init_weight() {};
  virtual void init_bias() {};

protected:
  Shape in_shape_;
  Shape out_shape_;
  Matrix in_;
  Matrix din_;
  Matrix w_;
  Matrix dw_;
  Matrix b_;
  Matrix db_;
  Optimizer<T> optimizer_;

};