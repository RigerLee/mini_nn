/**
 * @file network_impl.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief  the file for implementation of the network
 * @version 1.6.0
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 *
 */

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
xt::xarray<T> Network<T>::predict(const xt::xarray<T>& in) {
  Matrix out = in;
  for (auto& layer : layers_) {
    out = layer->forward(out);
  }
  return out;
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