/**
 * @file data_loader.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief 
 * @version 1.6.0
 * @date 2019-05-30
 * 
 * @copyright Copyright (c) 2019
 * 
 */

#pragma once

#include "xtensor-io/ximage.hpp"
#include "xtensor/xio.hpp"

#include <sys/time.h>
#include "common_header.hpp"
#include "utils.hpp"
/**
 * @brief the dataset class
 *
 * @tparam T
 * @details: read the images from binary form, 
 */
template <typename T>
class Dataset {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Dataset(bool shuffle = true);
  virtual ~Dataset() = default;

  /**
   * @brief read from MINST dataset
   * 
   * @param path 
   */
  void MNIST(const std::string& path);
  /**
   * @brief read images from binary form
   * 
   * @param image_file : the image
   * @param data        : the data
   */
  void read_bin_images(const std::string& image_file, xt::xarray<T>& data);
  /**
   * @brief read labels from binary form
   *
   * @param label_file: the labelled file
   * @param label     : the label of the file, from 1 to 9
   */
  void read_bin_labels(const std::string& label_file, xt::xarray<T>& label);
  /**
   * @brief load the image
   * 
   * @param mode : the ways we read
   * @param batch_size : the size of batch
   * @return std::vector<std::pair<Matrix, Matrix>> 
   */
  std::vector<std::pair<Matrix, Matrix>> loader(const std::string& mode,
                                                int batch_size = 1);
  /**
   * @brief normalize funtion
   *
   * @param mean:  average
   * @param stdev: std
   * @details Average the image according to the given mean and std
   */
  virtual void normalize(const xt::xarray<T>& mean, const xt::xarray<T>& stdev);

protected:
  bool shuffle_;
  Matrix train_data_;
  Matrix test_data_;
  Matrix train_label_;
  Matrix test_label_;
};

#include "data_loader_impl.hpp"