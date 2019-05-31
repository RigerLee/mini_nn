/**
 * @file pooling.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan
 * Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief
 * @version 1.6.0
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 *
 */

#pragma once

#include "layer_base.hpp"
/**
 * @brief the class for the maxpool
 *
 * @tparam T
 */
template <typename T>
class MaxPool2d : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  MaxPool2d() = default;
  virtual ~MaxPool2d() = default;

  /**
   * @brief Construct a new Max Pool 2d object
   *
   * @param kernel_size
   * @param stride
   * @param padding
   */
  MaxPool2d(size_t kernel_size, size_t stride = 0, size_t padding = 0);

  /**
   * @brief forward in the network
   *
   * @param in    the input matrix
   * @return      the output matrix
   */
  virtual Matrix forward(const Matrix& in) override;

  /**
   * @brief backward in the network
   *
   * @param dout    the derivative of output matrix
   * @return        the derivative of input matrix
   */
  virtual Matrix backward(const Matrix& dout) override;

protected:
  size_t kernel_size_;
  size_t padding_;
  size_t stride_;
};

#include "pooling_impl.hpp"