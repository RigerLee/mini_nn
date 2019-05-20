#pragma once

#include "layer.hpp"

template <typename T>
class ReLU : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  ReLU() = default;
  virtual ~ReLU() = default;

  virtual Matrix forward(const Matrix& in) override;
  virtual Matrix backward(const Matrix& dout) override;

};

#include "activation_impl.hpp"