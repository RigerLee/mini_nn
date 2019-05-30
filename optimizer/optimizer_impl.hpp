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

/**
 * @brief Construct a new SGD<T>::SGD object
 * 
 * @tparam T 
 * @param lr 
 * @param momentum 
 * @param weight_decay default is 0
 */

template <typename T>
SGD<T>::SGD(T lr, T momentum, T weight_decay) {
  this->lr_ = lr;
  momentum_ = momentum;
  weight_decay_ = weight_decay;
}

/**
 * @brief Stochastic gradient descent
 *
 * @tparam T
 * @param target
 * @param grad
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

{\displaystyle Q(w)={\frac {1}{n}}\sum _{i=1}^{n}Q_{i}(w),} {\displaystyle
Q(w)={\frac {1}{n}}\sum _{i=1}^{n}Q_{i}(w),} where the parameter {\displaystyle
w} w that minimizes {\displaystyle Q(w)} Q(w) is to be estimated. Each summand
function {\displaystyle Q_{i}} Q_{i} is typically associated with the
{\displaystyle i} i-th observation in the data set (used for training).

In classical statistics, sum-minimization problems arise in least squares and in
maximum-likelihood estimation (for independent observations). The general class
of estimators that arise as minimizers of sums are called M-estimators. However,
in statistics, it has been long recognized that requiring even local
minimization is too restrictive for some problems of maximum-likelihood
estimation.[2] Therefore, contemporary statistical theorists often consider
stationary points of the likelihood function (or zeros of its derivative, the
score function, and other estimating equations).

The sum-minimization problem also arises for empirical risk minimization. In
this case, {\displaystyle Q_{i}(w)} Q_{i}(w) is the value of the loss function
at {\displaystyle i} i-th example, and {\displaystyle Q(w)} Q(w) is the
empirical risk.

When used to minimize the above function, a standard (or "batch") gradient
descent method would perform the following iterations :

{\displaystyle w:=w-\eta \nabla Q(w)=w-\eta \sum _{i=1}^{n}\nabla Q_{i}(w)/n,}
{\displaystyle w:=w-\eta \nabla Q(w)=w-\eta \sum _{i=1}^{n}\nabla Q_{i}(w)/n,}
where {\displaystyle \eta } \eta  is a step size (sometimes called the learning
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
void SGD<T>::update(xt::xarray<T>& target, const xt::xarray<T>& grad) {
  target = momentum_ * target - this->lr_ * (grad + weight_decay_ * target);
}