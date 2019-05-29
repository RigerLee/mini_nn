/**
 * @file linear.hpp
 * @brief linear.hpp
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
class Linear : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Linear() = default;
  virtual ~Linear() = default;

  Linear(size_t in_dims, size_t out_dims);

  virtual Matrix forward(const Matrix& in) override;
  virtual Matrix backward(const Matrix& dout) override;

  // for init
  virtual size_t get_fan();

protected:
  Matrix in_reshape_;
  size_t in_dims_;
  size_t out_dims_;
};

#include "linear_impl.hpp"