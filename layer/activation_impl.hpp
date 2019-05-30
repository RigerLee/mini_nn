/**
 * @file activation_impl.hpp
 * @author RuiJian Li, YiFan Cao, YanPeng Hu
 * @brief the implentation for the header file of activation function
 * @version 1.6.0
 * @date 2019-05-30
 *
 * @copyright Copyright (c) 2019
 *
 */

/**
 * @brief Construct a new Re L U< T>:: Re L U object, the rectifier is an
activation function
 *
 * @tparam T
 * @details
 * In the context of artificial neural networks, the rectifier is an
activation function defined as the positive part of its argument:

{\displaystyle f(x)=x^{+}=\max(0,x)} {\displaystyle f(x)=x^{+}=\max(0,x)}

where x is the input to a neuron. This is also known as a ramp function and is
analogous to half-wave rectification in electrical engineering. This activation
function was first introduced to a dynamical network by Hahnloser et al. in 2000
with strong biological motivations and mathematical justifications. It has
been demonstrated for the first time in 2011 to enable better training of deeper
networks, compared to the widely-used activation functions prior to 2011,
e.g., the logistic sigmoid (which is inspired by probability theory; see
logistic regression) and its more practical counterpart, the hyperbolic
tangent. The rectifier is, as of 2017, the most popular activation function for
deep neural networks.
 */
template <typename T> ReLU<T>::ReLU() { this->layer_type_ = ACT; }

/**
 * @brief forward function
 *
 * @tparam T
 * @param in the input 
 * @return xt::xarray<T>
 */
template <typename T> xt::xarray<T> ReLU<T>::forward(const xt::xarray<T> &in) {
  this->in_ = in;
  Matrix out = xt::maximum(0, in);
  return out;
}

/**
 * @brief backward function in the network
 *
 * @tparam T
 * @param dout 
 * @return xt::xarray<T>
 */
template <typename T>
xt::xarray<T> ReLU<T>::backward(const xt::xarray<T> &dout) {
  this->din_ = xt::greater(this->in_, 0) * dout;
  return this->din_;
}