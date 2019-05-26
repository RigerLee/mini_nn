#pragma once

#include "common_header.hpp"

template <typename T>
class Optimizer {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Optimizer() = default;
  virtual ~Optimizer() = default;

  virtual void update(Matrix& target, const Matrix& grad) = 0;

protected:
  T lr_;
};