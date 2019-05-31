/**
 * @file linear.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief  the linear of the header file
 * @version 1.6.0
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 *
 */

#pragma once

#include "init.hpp"
#include "layer_base.hpp"
/**
 * @brief the layer class which inherits the linear class
 *
 * @tparam T
 */
template <typename T>
class Linear : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Linear() = default;
  virtual ~Linear() = default;


  /**
   * @brief Construct a new Linear object
   *
   * @param in_dims    input dimensions
   * @param out_dims   output dimensions
   */
  Linear(size_t in_dims, size_t out_dims);


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

  // for init
  /**
   * @brief Get the fan object
   *
   * @return size_t
   */
  virtual size_t get_fan();

protected:
  Matrix in_reshape_;
  size_t in_dims_;
  size_t out_dims_;
};

#include "linear_impl.hpp"