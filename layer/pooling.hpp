/**
 * @file pooling.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
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
/**
 * @brief Construct a new Max Pool 2d object
 * 
 */
  MaxPool2d() = default;
  /**
   * @brief Destroy the Max Pool 2d object
   * 
   */
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
 * @brief forward function in the network
 * 
 * @param in 
 * @return Matrix 
 */
  virtual Matrix forward(const Matrix& in) override;

/**
 * @brief backward function in the network
 * 
 * @param dout 
 * @return Matrix 
 */
  virtual Matrix backward(const Matrix& dout) override;

protected:
  size_t kernel_size_;
  size_t padding_;
  size_t stride_;
};

#include "pooling_impl.hpp"