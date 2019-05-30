/**
 * @file loss.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief define the loss for the network
 * @version 1.6.0
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 * 
 */
#pragma once

#include "common_header.hpp"
/**
 * @brief we define the type of loss of entropy
 * 
 */
enum LOSS_TYPE { CROSS_ENTROPY };
/**
 * @brief define the loss class
 * 
 * @tparam T 
 */
template <typename T>
class Loss {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;
/**
 * @brief Construct a new Loss object
 * 
 */
  Loss();
  /**
   * @brief Construct a new Loss object
   * 
   * @param loss_type 
   */
  Loss(LOSS_TYPE loss_type);
  /**
   * @brief Destroy the Loss object
   * 
   */
  virtual ~Loss() = default;
  /**
   * @brief Get the type object
   * 
   * @return LOSS_TYPE 
   */
  virtual LOSS_TYPE get_type() { return loss_type_; };
  /**
   * @brief Get the grad object
   * 
   * @return const Matrix& 
   */
  virtual const Matrix& get_grad() { return dscores_; };
  /**
   * @brief we use the cross entropy algorithm
   * 
   * @param scores 
   * @param target 
   * @return T 
   */
  virtual T CrossEntropyLoss(const Matrix& scores, const Matrix& target);

protected:
  Matrix scores_;
  Matrix dscores_;
  LOSS_TYPE loss_type_;
};

#include "loss_impl.hpp"