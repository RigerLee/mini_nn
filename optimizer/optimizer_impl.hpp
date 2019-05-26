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