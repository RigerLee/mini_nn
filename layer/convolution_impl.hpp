/**
 * @file convolution_impl.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief implementation for the convolution
 * @version 1.6.0
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#include <network.hpp>

/**
 * @brief Construct a new Conv 2d< T>:: Conv 2d object
 *
 * @tparam T
 * @param in_channels :   It refers to the input image that needs to be convolved. It
 * is required to be a Tensor with a shape such as [batch, in_height, in_width,
 * in_channels]. The specific meaning is [the number of pictures of a batch
 * during training, the height of the picture, the width of the image, the
 * number of image channels. ], note that this is a 4D Tensor,
 * @param out_channels:
 * @param kernel_size size of the kernel
 * @param stride The convolution step in each dimension of the image, this is a
 * one-dimensional vector, with length 4
 * @param padding  This value determines the different convolution methods
 */
template <typename T>
Conv2d<T>::Conv2d(size_t in_channels,
                  size_t out_channels,
                  size_t kernel_size,
                  size_t stride,
                  size_t padding) {
  this->layer_type_ = CONV;
  Shape W_shape = {out_channels, in_channels, kernel_size, kernel_size};
  Shape b_shape = {out_channels};

  this->in_channels_ = in_channels;
  this->out_channels_ = out_channels;
  this->kernel_size_ = kernel_size;
  this->padding_ = padding;
  this->stride_ = stride;
  // init?
  this->W_ = xt::zeros<T>(W_shape);  // Matrix(W_shape);
  this->b_ = xt::zeros<T>(b_shape);  // Matrix(b_shape);
  this->dW_ = Matrix(W_shape);
  this->db_ = Matrix(b_shape);
  this->net_ = nullptr;
  kaiming_normal(*this, "ReLU");
}
/**
 * @brief 
 * 
 * @tparam T 
 * @param in :input for the network
 * @return xt::xarray<T> 
 */
template <typename T>
xt::xarray<T> Conv2d<T>::forward(const xt::xarray<T>& in) {
  if (in.shape().size() != 4) {
    std::cerr << "Input shape: (N, C, H, W) " << std::endl;
    exit(0);
  }

  this->in_ = in;
  size_t H, W, H_out, W_out;
  H = in.shape(2);
  W = in.shape(3);
  /**
   * @brief prepare for output
   *
   */
  H_out = 1 + (H + 2 * this->padding_ - this->kernel_size_) / this->stride_;
  W_out = 1 + (W + 2 * this->padding_ - this->kernel_size_) / this->stride_;
  Matrix out = xt::zeros<T>({in.shape(0), this->out_channels_, H_out, W_out});

  // conv2d, maybe omp?
  Matrix in_pad = xt::pad(in, {{0, 0},
                               {0, 0},
                               {this->padding_, this->padding_},
                               {this->padding_, this->padding_}});

  for (size_t i = 0; i < H_out; ++i) {
    for (size_t j = 0; j < W_out; ++j) {
      auto in_pad_part = xt::view(
        in_pad, xt::all(), xt::all(),
        xt::range(i * this->stride_, i * this->stride_ + this->kernel_size_),
        xt::range(j * this->stride_, j * this->stride_ + this->kernel_size_));

      for (size_t k = 0; k < this->out_channels_; ++k) {
        auto out_part = xt::view(out, xt::all(), k, i, j);
        out_part = xt::sum(
          in_pad_part * xt::view(this->W_, k, xt::all(), xt::all(), xt::all()),
          {1, 2, 3});
      }
    }
  }
  out = out + xt::view(xt::view(this->b_, xt::all()), xt::newaxis(), xt::all(),
                       xt::newaxis(), xt::newaxis());
  return out;
}
/**
 * @brief 
 * 
 * @tparam T 
 * @param dout : the backward input
 * @return xt::xarray<T> 
 */
template <typename T>
xt::xarray<T> Conv2d<T>::backward(const xt::xarray<T>& dout) {
  size_t H, W, H_out, W_out;
  H = this->in_.shape(2);
  W = this->in_.shape(3);

  H_out = 1 + (H + 2 * this->padding_ - this->kernel_size_) / this->stride_;
  W_out = 1 + (W + 2 * this->padding_ - this->kernel_size_) / this->stride_;
  // db is easy to get
  this->db_ = xt::sum(dout, {0, 2, 3});

  Matrix in_pad = xt::pad(this->in_, {{0, 0},
                                      {0, 0},
                                      {this->padding_, this->padding_},
                                      {this->padding_, this->padding_}});
  Matrix din_pad = xt::zeros_like(in_pad);
  this->dW_ = xt::zeros_like(this->W_);

  for (size_t i = 0; i < H_out; ++i) {
    for (size_t j = 0; j < W_out; ++j) {
      auto in_pad_part = xt::view(
        in_pad, xt::all(), xt::all(),
        xt::range(i * this->stride_, i * this->stride_ + this->kernel_size_),
        xt::range(j * this->stride_, j * this->stride_ + this->kernel_size_));
      // dW_
      for (size_t k = 0; k < this->out_channels_; ++k) {
        auto dW_part = xt::view(this->dW_, k, xt::all(), xt::all(), xt::all());

        dW_part += xt::sum(
          in_pad_part * xt::view(xt::view(dout, xt::all(), k, i, j), xt::all(),
                                 xt::newaxis(), xt::newaxis(), xt::newaxis()),
          {0});
      }
      // din_pad
      for (size_t n = 0; n < this->in_.shape(0); ++n) {
        auto din_pad_part = xt::view(
          din_pad, n, xt::all(),
          xt::range(i * this->stride_, i * this->stride_ + this->kernel_size_),
          xt::range(j * this->stride_, j * this->stride_ + this->kernel_size_));

        din_pad_part += xt::sum(
          this->W_ * xt::view(xt::view(dout, n, xt::all(), i, j), xt::all(),
                              xt::newaxis(), xt::newaxis(), xt::newaxis()),
          {0});
      }
    }
  }

  this->din_ =
    xt::view(din_pad, xt::all(), xt::all(),
             xt::range(this->padding_, din_pad.shape(2) - this->padding_),
             xt::range(this->padding_, din_pad.shape(3) - this->padding_));
  /**
   * @brief only update if net is already set
   *
   */
  if (this->net_) {
    this->net_->get_optimizer()->update(this->W_, this->dW_);
    this->net_->get_optimizer()->update(this->b_, this->db_);
  }

  return this->din_;
}
/**
 * @brief get the fan of Conv2d
 * 
 * @tparam T 
 * @return size_t 
 */
template <typename T>
size_t Conv2d<T>::get_fan() {
  return this->W_.shape(1) * this->W_.shape(2) * this->W_.shape(3);
}
