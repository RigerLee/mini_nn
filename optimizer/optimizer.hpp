#pragma once

#include "optimizer_base.hpp"

template <typename T>
class SGD : public Optimizer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  SGD(T lr = 0.1, T momentum = 1., T weight_decay = 0.);
  virtual ~SGD() = default;

  virtual void update(Matrix& target, const Matrix& grad) override;

protected:
  T momentum_;
  T weight_decay_;
};

#include "optimizer_impl.hpp"