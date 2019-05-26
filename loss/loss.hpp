#pragma once

#include "common_header.hpp"

enum LOSS_TYPE { CROSS_ENTROPY };

template <typename T>
class Loss {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Loss();
  Loss(LOSS_TYPE loss_type);
  virtual ~Loss() = default;

  virtual LOSS_TYPE get_type() { return loss_type_; };
  virtual const Matrix& get_grad() { return dscores_; };

  virtual T CrossEntropyLoss(const Matrix& scores, const Matrix& target);

protected:
  Matrix scores_;
  Matrix dscores_;
  LOSS_TYPE loss_type_;
};

#include "loss_impl.hpp"