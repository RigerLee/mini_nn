/**
 * @file optimizer.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan
 * Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief the header file of optimizer
 * @version 1.6.0
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 *
 */
#pragma once

#include "optimizer_base.hpp"
/**
 * @brief Stochastic gradient descent
 *
 * @tparam T
 * @details  Stochastic gradient descent (often abbreviated SGD) is an iterative
method
 * for optimizing an objective function with suitable smoothness properties
 * (e.g. differentiable or subdifferentiable). It is called stochastic because
 * the method uses randomly selected (or shuffled) samples to evaluate the
 * gradients, hence SGD can be regarded as a stochastic approximation of
 * gradient descent optimization. The ideas can be traced back[1] at least to
 * the 1951 article titled "A Stochastic Approximation Method" by Herbert
 * Robbins and Sutton Monro, who proposed with detailed analysis a root-finding
 * method now called the Robbins-Monro algorithm .
 *
 * Both statistical estimation and machine learning consider the problem of
minimizing an objective function that has the form of a sum:

\f[ Q(w)={\frac {1}{n}}\sum _{i=1}^{n}Q_{i}(w), \f]

where the parameter \f$ w \f$ that minimizes \f$ Q(w) \f$ is to be estimated.
Each summand function \f$ Q_i \f$ is typically associated with the \f$ i \f$-th
observation in the data set (used for training).

In classical statistics, sum-minimization problems arise in least squares and in
maximum-likelihood estimation (for independent observations). The general class
of estimators that arise as minimizers of sums are called M-estimators. However,
in statistics, it has been long recognized that requiring even local
minimization is too restrictive for some problems of maximum-likelihood
estimation.[2] Therefore, contemporary statistical theorists often consider
stationary points of the likelihood function (or zeros of its derivative, the
score function, and other estimating equations).

The sum-minimization problem also arises for empirical risk minimization. In
this case, \f$ Q_i(w) \f$ is the value of the loss function
at \f$ i \f$-th example, and \f$ Q(w) \f$ is the
empirical risk.

When used to minimize the above function, a standard (or "batch") gradient
descent method would perform the following iterations :

\f[ w:=w-\eta \nabla Q(w)=w-\eta \sum _{i=1}^{n}\nabla Q_{i}(w)/n, \f]

where \f$ \eta \f$ is a step size (sometimes called the learning
rate in machine learning).

In many cases, the summand functions have a simple form that enables inexpensive
evaluations of the sum-function and the sum gradient. For example, in
statistics, one-parameter exponential families allow economical
function-evaluations and gradient-evaluations.

However, in other cases, evaluating the sum-gradient may require expensive
evaluations of the gradients from all summand functions. When the training set
is enormous and no simple formulas exist, evaluating the sums of gradients
becomes very expensive, because evaluating the gradient requires evaluating all
the summand functions' gradients. To economize on the computational cost at
every iteration, stochastic gradient descent samples a subset of summand
functions at every step. This is very effective in the case of large-scale
machine learning problems.[3]

 */
template <typename T>
class SGD : public Optimizer<T> {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;
  /**
   * @brief Construct a new SGD object
   *
   * @param lr init with 0.
   * @param momentum init with 1
   * @param weight_decay default is 0
   */
  SGD(T lr = 0.1, T momentum = 1., T weight_decay = 0.);

  /**
   * @brief Destroy the SGD object
   *
   */
  virtual ~SGD() = default;

  /**
   * @brief update the network
   *
   * @param target
   * @param grad
   */
  virtual void update(Matrix& target, const Matrix& grad) override;

protected:
  T momentum_;
  T weight_decay_;
};

#include "optimizer_impl.hpp"