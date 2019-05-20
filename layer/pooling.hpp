#pragma once

#include "layer.hpp"

template <typename T>
class MaxPool2d : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  MaxPool2d() = default;
  virtual ~MaxPool2d() = default;

  virtual Matrix forward(const Matrix& in) override;
  virtual Matrix backward(const Matrix& dout) override;

};

#include "pooling_impl.hpp"