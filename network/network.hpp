#pragma once

#include <list>
#include "activation.hpp"
#include "convolution.hpp"
#include "linear.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "pooling.hpp"

template <typename T>
class Network {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Network() = default;
  virtual ~Network() = default;

  // add layers, only pointer and reference is league for abstract class
  Network<T>& operator<<(Layer<T>* layer);
  Network<T>& operator<<(Layer<T>& layer);
  // add loss func
  Network<T>& operator<<(Loss<T>* loss);
  Network<T>& operator<<(Loss<T>& loss);

  virtual Optimizer<T>* get_optimizer() { return optimizer_; };
  virtual void set_optimizer(Optimizer<T>* opt) { optimizer_ = opt; };

  virtual Matrix forward(const Matrix& in, const Matrix& target);
  virtual void backward();

protected:
  std::list<Layer<T>*> layers_;
  Loss<T>* loss_;
  Optimizer<T>* optimizer_;
};

#include "network_impl.hpp"