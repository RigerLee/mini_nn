/**
 * @file network.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan
 * Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
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
   * @param layer   the pointer to the layer
   * @return        self, so that << can be continuously used
   */
  Network<T>& operator<<(Layer<T>* layer);
  /**
   * @brief Polymorphism: define the layer for the reference
   *
   * @tparam T
   * @param layer   the reference of the layer
   * @return        self, so that << can be continuously used
   */
  Network<T>& operator<<(Layer<T>& layer);
  // add loss func
  /**
   * @brief Polymorphism: define the loss function for the pointer
   *
   * @tparam T
   * @param loss    the pointer to the loss
   * @return        self, so that << can be continuously used
   */
  Network<T>& operator<<(Loss<T>* loss);
  /**
   * @brief Polymorphism: define the loss function for the refernce
   *
   * @tparam T
   * @param loss    the reference of the loss
   * @return        self, so that << can be continuously used
   */
  Network<T>& operator<<(Loss<T>& loss);

  /**
   * @brief Get the optimizer object
   *
   * @return   the pointer to the optimizer object
   */
  virtual Optimizer<T>* get_optimizer() { return optimizer_; };
  /**
   * @brief Set the optimizer object
   *
   * @param opt   the pointer to optimizer object
   */
  virtual void set_optimizer(Optimizer<T>* opt) { optimizer_ = opt; };
  /**
   * @brief predict the result given an input
   *
   * @param in   the input matrix
   * @return     the predicted result
   */
  virtual Matrix predict(const Matrix& in);

  /**
   * @brief forward function in the network
   *
   * @param in       the input matrix
   * @param target   the labels for the inputs
   * @return         the loss
   */
  virtual Matrix forward(const Matrix& in, const Matrix& target);

  /**
   * @brief backward function in the network
   *
   */
  virtual void backward();

protected:
  std::list<Layer<T>*> layers_;
  Loss<T>* loss_;
  Optimizer<T>* optimizer_;
};

#include "network_impl.hpp"