/**
 * @file network.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief the definition for the network
 * @version 1.6.0
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once

#include <list>
#include "activation.hpp"
#include "convolution.hpp"
#include "linear.hpp"
#include "loss.hpp"
#include "optimizer.hpp"
#include "pooling.hpp"
/**
 * @brief the defintion for the network
 * 
 * @tparam T 
 */
template <typename T>
class Network {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Network() = default;
  virtual ~Network() = default;

  // add layers, only pointer and reference is league for abstract class
  /**
   * @brief Polymorphism: define the layer for the pointer
   *
   * @tparam T
   * @param layer
   * @return Network<T>&
   */
  Network<T>& operator<<(Layer<T>* layer);
  /**
   * @brief Polymorphism: define the layer for the refernce
   *
   * @tparam T
   * @param layer
   * @return Network<T>&
   */
  Network<T>& operator<<(Layer<T>& layer);
  // add loss func
  /**
   * @brief Polymorphism: define the loss function for the pointer
   *
   * @tparam T
   * @param layer
   * @return Network<T>&
   */
  Network<T>& operator<<(Loss<T>* loss);
  /**
   * @brief Polymorphism: define the loss function for the refernce
   *
   * @tparam T
   * @param layer
   * @return Network<T>&
   */
  Network<T>& operator<<(Loss<T>& loss);

  /**
   * @brief Get the optimizer object
   * 
   * @return Optimizer<T>* 
   */
  virtual Optimizer<T>* get_optimizer() { return optimizer_; };
  /**
   * @brief Set the optimizer object
   * 
   * @param opt 
   */
  virtual void set_optimizer(Optimizer<T>* opt) { optimizer_ = opt; };
  /**
   * @brief predict the label of the image
   * 
   * @param in 
   * @return Matrix 
   */
  virtual Matrix predict(const Matrix& in);
  /**
   * @brief forward in the network
   * 
   * @param in 
   * @param target 
   * @return Matrix 
   */
  virtual Matrix forward(const Matrix& in, const Matrix& target);
  /**
   * @brief backward in the network
   * 
   */
  virtual void backward();

protected:
  std::list<Layer<T>*> layers_;
  Loss<T>* loss_;
  Optimizer<T>* optimizer_;
};

#include "network_impl.hpp"