#pragma once

#include "layer.hpp"

template <typename T>
class ReLU : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  ReLU() = default;
  virtual ~ReLU() = default;

  virtual Matrix forward() override;
  virtual Matrix backward() override;

};

#include "activation_impl.hpp"