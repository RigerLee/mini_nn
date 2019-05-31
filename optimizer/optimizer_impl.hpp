/**
 * @file optimizer_impl.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan
 * Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief implementation for the optimizer
 * @version 1.6.0
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 *
 */

template <typename T>
SGD<T>::SGD(T lr, T momentum, T weight_decay) {
  this->lr_ = lr;
  momentum_ = momentum;
  weight_decay_ = weight_decay;
}

template <typename T>
void SGD<T>::update(xt::xarray<T>& target, const xt::xarray<T>& grad) {
  target = momentum_ * target - this->lr_ * (grad + weight_decay_ * target);
}