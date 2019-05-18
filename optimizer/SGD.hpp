#pragma once

#include "optimizer.hpp"

template <typename T>
class SGD : public Optimizer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  SGD() = default;
  virtual ~SGD() = default;

  virtual Matrix update() override;

protected:

};

#include "SGD_impl.hpp"