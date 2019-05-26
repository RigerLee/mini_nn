#pragma once

#include "layer_base.hpp"

template <typename T>
class MaxPool2d : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  MaxPool2d() = default;
  virtual ~MaxPool2d() = default;

  MaxPool2d(size_t kernel_size, size_t stride = 0, size_t padding = 0);

  virtual Matrix forward(const Matrix& in) override;
  virtual Matrix backward(const Matrix& dout) override;

protected:
  size_t kernel_size_;
  size_t padding_;
  size_t stride_;
};

#include "pooling_impl.hpp"