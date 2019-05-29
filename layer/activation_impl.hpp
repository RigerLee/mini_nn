/**
 * \file activation_impl.hpp
 * \brief activation_impl.hpp
 * \details head file
 * \mainpage mini_nn
 * \author RuiJian Li, YiFan Cao, YanPeng Hu
 * \email lirj@shanghaitech.edu.cn,caoyf@shanghaitech.edu.cn,huyp@shanghaitech.edu.cn
 * \version 1.6.0
 * \date 2019-05-26
 */

template <typename T> ReLU<T>::ReLU() { this->layer_type_ = ACT; }

template <typename T> xt::xarray<T> ReLU<T>::forward(const xt::xarray<T> &in) {
  this->in_ = in;
  Matrix out = xt::maximum(0, in);
  return out;
}

template <typename T>
xt::xarray<T> ReLU<T>::backward(const xt::xarray<T> &dout) {
  this->din_ = xt::greater(this->in_, 0) * dout;
  return this->din_;
}