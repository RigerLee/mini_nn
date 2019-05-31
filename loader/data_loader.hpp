/**
 * @file data_loader.hpp
 * @author RuiJian Li(lirj@shanghaitech.edu.cn), YiFan
 * Cao(caoyf@shanghaitech.edu.cn), YanPeng Hu(huyp@shanghaitech.edu.cn)
 * @brief load data
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
 * @brief Class of the dataset
 *
 * @tparam T
 * @param shuffle the data
 * @details Shuffling data serves the purpose of reducing variance and making
sure that models remain general and overfit less.

The obvious case where you'd shuffle your data is if your data is sorted by
their class/target. Here, you will want to shuffle to make sure that your
training/test/validation sets are representative of the overall distribution of
the data.

For batch gradient descent, the same logic applies. The idea behind batch
gradient descent is that by calculating the gradient on a single batch, you will
usually get a fairly good estimate of the "true" gradient. That way, you save
computation time by not having to calculate the "true" gradient over the entire
dataset every time.

You want to shuffle your data after each epoch because you will always have the
risk to create batches that are not representative of the overall dataset, and
therefore, your estimate of the gradient will be off. Shuffling your data after
each epoch ensures that you will not be "stuck" with too many bad batches.

In regular stochastic gradient descent, when each batch has size 1, you still
want to shuffle your data after each epoch to keep your learning general.
Indeed, if data point 17 is always used after data point 16, its own gradient
will be biased with whatever updates data point 16 is making on the model. By
shuffling your data, you ensure that each data point creates an "independent"
change on the model, without being biased by the same points before them.
 */
template <typename T>
class Dataset {
public:
  typedef xt::xarray<T> Matrix;
  typedef typename Matrix::shape_type Shape;

  Dataset(bool shuffle = true);
  virtual ~Dataset() = default;

  /**
   * @brief read from MNIST dataset
   *
   * @param path   the pathname of the MNIST dataset
   */
  void MNIST(const std::string& path);

  /**
   * @brief read images from binary form
   *
   * @param image_file   the image file name
   * @param data         read the image to this matrix
   */
  void read_bin_images(const std::string& image_file, xt::xarray<T>& data);

  /**
   * @brief read labels from binary form
   *
   * @param label_file   the pathname of the labeled file
   * @param label        the label of the file, from 1 to 9
   */
  void read_bin_labels(const std::string& label_file, xt::xarray<T>& label);

  /**
   * @brief load the image
   *
   * @param mode         the ways we read
   * @param batch_size   the size of batch
   * @return a vector of pairs of the input matrix together with the label
   */
  std::vector<std::pair<Matrix, Matrix>> loader(const std::string& mode,
                                                int batch_size = 1);

  /**
   * @brief normalize function for the train and test data
   *
   * @tparam T
   * @param mean    mean
   * @param stdev   standard deviation
   * @details Average the image according to the given mean and standard
   * deviation, by calling data_normalize.
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
