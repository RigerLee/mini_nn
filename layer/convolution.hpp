/**
 * @file convolution.hpp
 * @brief convolution.hpp
 * @details head file
 * @mainpage mini_nn
 * @author RuiJian Li, YiFan Cao, YanPeng Hu
 * @email lirj@shanghaitech.edu.cn,
 * caoyf@shanghaitech.edu.cn,huyp@shanghaitech.edu.cn
 * @version 1.6.0
 * @date 2019-05-26
 */

#pragma once

#include "init.hpp"
#include "layer_base.hpp"

template <typename T>
class Conv2d : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Conv2d() = default;
  virtual ~Conv2d() = default;

  Conv2d(size_t in_channels,
         size_t out_channels,
         size_t kernel_size = 3,
         size_t stride = 1,
         size_t padding = 0);

  virtual Matrix forward(const Matrix& in) override;
  virtual Matrix backward(const Matrix& dout) override;

  // for init
  virtual size_t get_fan();

protected:
  size_t in_channels_;
  size_t out_channels_;
  size_t kernel_size_;
  size_t padding_;
  size_t stride_;
};

#include "convolution_impl.hpp"