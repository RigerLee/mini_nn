/**
 * @file pooling_impl.hpp
 * @brief the implementation of the pooling
 * @details head file
 * @mainpage mini_nn
 * @author RuiJian Li, YiFan Cao, YanPeng Hu
 * @email lirj@shanghaitech.edu.cn,
 * caoyf@shanghaitech.edu.cn,huyp@shanghaitech.edu.cn
 * @version 1.6.0
 * @date 2019-05-26
 */

template <typename T>
MaxPool2d<T>::MaxPool2d(size_t kernel_size, size_t stride, size_t padding) {
  this->layer_type_ = POOL;
  // default stride is kernel_size for pooling
  if (stride == 0) stride = kernel_size;
  this->kernel_size_ = kernel_size;
  this->padding_ = padding;
  this->stride_ = stride;
}

template <typename T>
xt::xarray<T> MaxPool2d<T>::forward(const xt::xarray<T>& in) {
  if (in.shape().size() != 4) {
    std::cerr << "Input shape: (N, C, H, W) " << std::endl;
    exit(0);
  }

  this->in_ = in;
  size_t H, W, H_out, W_out;
  H = in.shape(2);
  W = in.shape(3);
  // prepare for output
  H_out = 1 + (H + 2 * this->padding_ - this->kernel_size_) / this->stride_;
  W_out = 1 + (W + 2 * this->padding_ - this->kernel_size_) / this->stride_;
  Matrix out = xt::zeros<T>({in.shape(0), in.shape(1), H_out, W_out});

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

      auto out_part = xt::view(out, xt::all(), xt::all(), i, j);
      out_part = xt::amax(in_pad_part, {2, 3});
    }
  }

  return out;
}

template <typename T>
xt::xarray<T> MaxPool2d<T>::backward(const xt::xarray<T>& dout) {
  size_t H, W, H_out, W_out;
  H = this->in_.shape(2);
  W = this->in_.shape(3);

  H_out = 1 + (H + 2 * this->padding_ - this->kernel_size_) / this->stride_;
  W_out = 1 + (W + 2 * this->padding_ - this->kernel_size_) / this->stride_;

  Matrix in_pad = xt::pad(this->in_, {{0, 0},
                                      {0, 0},
                                      {this->padding_, this->padding_},
                                      {this->padding_, this->padding_}});

  Matrix din_pad = xt::zeros_like(in_pad);

  for (size_t i = 0; i < H_out; ++i) {
    for (size_t j = 0; j < W_out; ++j) {
      auto in_pad_part = xt::view(
        in_pad, xt::all(), xt::all(),
        xt::range(i * this->stride_, i * this->stride_ + this->kernel_size_),
        xt::range(j * this->stride_, j * this->stride_ + this->kernel_size_));

      auto max_mask = xt::amax(in_pad_part, {2, 3});
      auto binary_mask = xt::equal(
        in_pad_part,
        xt::view(max_mask, xt::all(), xt::all(), xt::newaxis(), xt::newaxis()));

      auto din_pad_part = xt::view(
        din_pad, xt::all(), xt::all(),
        xt::range(i * this->stride_, i * this->stride_ + this->kernel_size_),
        xt::range(j * this->stride_, j * this->stride_ + this->kernel_size_));

      din_pad_part +=
        binary_mask * xt::view(xt::view(dout, xt::all(), xt::all(), i, j),
                               xt::all(), xt::all(), xt::newaxis(),
                               xt::newaxis());
    }
  }

  this->din_ =
    xt::view(din_pad, xt::all(), xt::all(),
             xt::range(this->padding_, din_pad.shape(2) - this->padding_),
             xt::range(this->padding_, din_pad.shape(3) - this->padding_));

  return this->din_;
}
