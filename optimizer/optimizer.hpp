#pragma once

#include <iostream>
#include "xtensor-blas/xlinalg.hpp"
#include "xtensor/xarray.hpp"
#include "xtensor/xio.hpp"
#include "xtensor/xmath.hpp"
#include "xtensor/xpad.hpp"
#include "xtensor/xrandom.hpp"
#include "xtensor/xview.hpp"

template <typename T>
class Optimizer {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Optimizer() = default;
  virtual ~Optimizer() = default;

  virtual Matrix update() { return Matrix(); };

protected:
};