#pragma once

#include "layer.hpp"

template <typename T>
class Linear : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Linear() = default;
  virtual ~Linear() = default;

  Linear(size_t in_dims, size_t out_dims);

  virtual Matrix forward(const Matrix& in) override;
  virtual Matrix backward(const Matrix& dout) override;

  virtual void init_weight() override {};
  virtual void init_bias() override {};

protected:
  Matrix in_reshape_;
  size_t in_dims_;
  size_t out_dims_;
};

#include "linear_impl.hpp"