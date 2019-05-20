#pragma once

#include "layer.hpp"

template <typename T>
class Conv2d : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Conv2d() = default;
  virtual ~Conv2d() = default;

  Conv2d(size_t in_channels,
         size_t out_channels=16,
         size_t kernel_size=3,
         size_t stride=1,
         size_t padding=1);

  virtual Matrix forward(const Matrix& in) override;
  virtual Matrix backward(const Matrix& dout) override;

  virtual void init_weight() override {};
  virtual void init_bias() override {};
protected:
  size_t in_channels_;
  size_t out_channels_;
  size_t kernel_size_;
  size_t padding_;
  size_t stride_;
};

#include "convolution_impl.hpp"