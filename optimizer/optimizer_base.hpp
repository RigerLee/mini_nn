/**
 * @file optimizer_base.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan
 * Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief the header file for the optimizer_base
 * @version 1.6.0
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 *
 */
#pragma once

#include "common_header.hpp"

/**
 * @brief the class for the optimizer
 *
 * @tparam T
 * @details For the optimization algorithm, the goal of the optimization is the
 * parameter θ (which is a set, θ1, θ2, θ3 ...) in the network model. The
 * objective function is the loss function L = 1/N ∑ Li (for each sample loss
 * function) Add the average value). The loss function L variable is θ, where
 * the parameter in L is the entire training set. In other words, the objective
 * function (loss function) is determined by the entire training set. If the
 * training set is different, the loss function image is different. So why can't
 * I optimize the saddle point/local minimum point in mini-batch? Because at
 * these points, the gradient of L to θ is zero. In other words, the partial
 * derivative of each component of θ is taken into the complete set of the
 * training set, and the derivative is zero. For SGD/MBGD, the loss function
 * used each time is only determined by the data of this small batch, and the
 * function image is different from the real ensemble loss function, so the
 * gradient of the solution also contains a certain randomness at the saddle
 * point. Or when the local minimum point is oscillating, because at this point,
 * if the training set is brought into BGD, the optimization will stop. If it is
 * mini-batch or SGD, the gradient will be different each time. , there will be
 * shocks, jumping back and forth.
 *
 */
template <typename T> class Optimizer {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  /**
   * @brief Construct a new Optimizer object
   *
   */
  Optimizer() = default;

  /**
   * @brief Destroy the Optimizer object
   *
   */
  virtual ~Optimizer() = default;

  /**
   * @brief update the network
   * 
   * @param target : the aim 
   * @param grad : gradient
   */
  virtual void update(Matrix &target, const Matrix &grad) = 0;

protected:
  T lr_;
};