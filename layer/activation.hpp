/**
 * @file activation.hpp
 * @brief activation.hpp
 * @details head file
 * @mainpage mini_nn
 * @author RuiJian Li, YiFan Cao, YanPeng Hu
 * @email lirj@shanghaitech.edu.cn,
 * caoyf@shanghaitech.edu.cn,huyp@shanghaitech.edu.cn
 * @version 1.6.0
 * @date 2019-05-26
 */

#pragma once

#include "layer_base.hpp"

template <typename T>
class ReLU : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  ReLU();
  virtual ~ReLU() = default;

  virtual Matrix forward(const Matrix& in) override;
  virtual Matrix backward(const Matrix& dout) override;
};

#include "activation_impl.hpp"