#pragma once

#include "layer.hpp"

template <typename T>
class Conv2d : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Conv2d() = default;
  virtual ~Conv2d() = default;

  virtual Matrix forward() override;
  virtual Matrix backward() override;

  virtual void init_weight() override;
  virtual void init_bias() override;
};

#include "convolution_impl.hpp"