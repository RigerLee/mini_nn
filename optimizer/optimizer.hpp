#pragma once

#include <iostream>

template <typename T>
class Optimizer {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Optimizer() = default;
  virtual ~Optimizer() = default;

  virtual Matrix update() {return Matrix();};

protected:

};