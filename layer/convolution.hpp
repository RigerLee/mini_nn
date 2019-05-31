/**
 * @file convolution.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan
 * Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief  the attribute of the convolution
 * @version 1.6.0
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 *
 */

#pragma once

#include "init.hpp"
#include "layer_base.hpp"

template <typename T>
/**
 * @brief Computes a 2-D convolution given 4-D input and filter tensors
 * @details Given an input tensor of shape [batch, in_height, in_width,
in_channels] and a filter / kernel tensor of shape [filter_height, filter_width,
in_channels, out_channels], this op performs the following:

Flattens the filter to a 2-D matrix with shape [filter_height * filter_width *
in_channels, output_channels]. Extracts image patches from the input tensor to
form a virtual tensor of shape [batch, out_height, out_width, filter_height *
filter_width * in_channels]. For each patch, right-multiplies the filter matrix
and the image patch vector.
 */
class Conv2d : public Layer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Conv2d() = default;
  virtual ~Conv2d() = default;

  /**
   * @brief Construct a new Conv2d object
   *
   * @tparam T
   * @param in_channels
   * It refers to the input image that needs to be
   * convolved. It is required to be a Tensor with a shape such as [batch,
   * in_height, in_width, in_channels]. The specific meaning is [the number of
   * pictures of a batch during training, the height of the picture, the width
   * of the image, the number of image channels]. Note that this is a 4D
   * tensor.
   * @param out_channels
   * the number of output channels
   * @param kernel_size
   * size of the kernel
   * @param stride
   * The convolution step in each dimension of the image, this
   * is a one-dimensional vector, with length 4
   * @param padding
   * This value determines the different convolution methods
   */
  Conv2d(size_t in_channels,
         size_t out_channels,
         size_t kernel_size = 3,
         size_t stride = 1,
         size_t padding = 0);

  /**
   * @brief forward function in the network
   *
   * @param in     the input matrix
   * @return       the output matrix
   * @details
   */
  virtual Matrix forward(const Matrix& in) override;

  /**
   * @brief backward function in the network
   *
   * @param dout     the derivative of output matrix
   * @return         the derivative of input matrix
   */
  virtual Matrix backward(const Matrix& dout) override;

  /**
   * @brief Get the fan object
   *
   * @return size_t
   */
  virtual size_t get_fan();

protected:
  size_t in_channels_;
  size_t out_channels_;
  size_t kernel_size_;
  size_t padding_;
  size_t stride_;
};

#include "convolution_impl.hpp"