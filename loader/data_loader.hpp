/**
 * @file data_loader.hpp
 * @brief data_loader.hpp
 * @details head file
 * @mainpage mini_nn
 * @author RuiJian Li, YiFan Cao, YanPeng Hu
 * @email lirj@shanghaitech.edu.cn,
 * caoyf@shanghaitech.edu.cn,huyp@shanghaitech.edu.cn
 * @version 1.6.0
 * @date 2019-05-26
 */

#pragma once

#include "xtensor-io/ximage.hpp"
#include "xtensor/xio.hpp"

#include <sys/time.h>
#include "common_header.hpp"
#include "utils.hpp"

template <typename T>
class Dataset {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Dataset(bool shuffle = true);
  virtual ~Dataset() = default;

  void MNIST(const std::string& path);
  void read_bin_images(const std::string& image_file, xt::xarray<T>& data);
  void read_bin_labels(const std::string& label_file, xt::xarray<T>& label);
  std::vector<std::pair<Matrix, Matrix>> loader(const std::string& mode,
                                                int batch_size = 1);
  virtual void normalize(const xt::xarray<T>& mean, const xt::xarray<T>& stdev);

protected:
  bool shuffle_;
  Matrix train_data_;
  Matrix test_data_;
  Matrix train_label_;
  Matrix test_label_;
};

#include "data_loader_impl.hpp"
