template <typename T>
Network<T>& Network<T>::operator<<(Layer<T>* layer) {
  layer->set_network(this);
  layers_.push_back(layer);
  return *this;
}

template <typename T>
Network<T>& Network<T>::operator<<(Layer<T>& layer) {
  layer->set_network(this);
  layers_.push_back(&layer);
  return *this;
}

template <typename T>
Network<T>& Network<T>::operator<<(Loss<T>* loss) {
  loss_ = loss;
  return *this;
}

template <typename T>
Network<T>& Network<T>::operator<<(Loss<T>& loss) {
  loss_ = &loss;
  return *this;
}

template <typename T>
xt::xarray<T> Network<T>::forward(const xt::xarray<T>& in,
                                  const xt::xarray<T>& target) {
  Matrix out = in;
  for (auto& layer : layers_) {
    out = layer->forward(out);
  }
  if (loss_->get_type() == CROSS_ENTROPY) {
    out = loss_->CrossEntropyLoss(out, target);
  }
  return out;
}

template <typename T>
void Network<T>::backward() {
  Matrix din = loss_->get_grad();
  for (auto layer_iter = layers_.rbegin(); layer_iter != layers_.rend();
       ++layer_iter) {
    din = (*layer_iter)->backward(din);
  }
}