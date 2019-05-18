#pragma once

#include "layer.hpp"

template <typename T>
class Linear : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Linear() = default;
  virtual ~Linear() = default;

  virtual Matrix forward() override;
  virtual Matrix backward() override;

  virtual void init_weight() override;
  virtual void init_bias() override;

};

#include "linear_impl.hpp"